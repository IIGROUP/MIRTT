import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from transformers import BertPreTrainedModel, BertConfig, BertModel

import numpy as np
import math



# ---------------------------------------------------
# SAN
# ---------------------------------------------------
class StackedAttention(nn.Module):
    def __init__(self, num_stacks, img_feat_size, ques_feat_size, att_size, output_size, drop_ratio):
        super(StackedAttention, self).__init__()

        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size
        self.att_size = att_size
        self.output_size = output_size
        self.drop_ratio = drop_ratio
        self.num_stacks = num_stacks
        self.layers = nn.ModuleList()

        self.dropout = nn.Dropout(drop_ratio)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.fc11 = nn.Linear(ques_feat_size, att_size, bias=True)
        self.fc12 = nn.Linear(img_feat_size, att_size, bias=False)
        self.fc13 = nn.Linear(att_size, 1, bias=True)

        self.fc14 = nn.Linear(ques_feat_size, att_size, bias=True)
        self.fc15 = nn.Linear(img_feat_size, att_size, bias=False)

        for stack in range(num_stacks - 1):
            self.layers.append(nn.Linear(att_size, att_size, bias=True))
            self.layers.append(nn.Linear(img_feat_size, att_size, bias=False))
            self.layers.append(nn.Linear(att_size, 1, bias=True))

    def forward(self, img_feat, ques_feat, v_mask=False):

        # Batch size
        B = ques_feat.size(0)

        # Stack 1
        ques_emb = self.fc11(ques_feat)
        img_emb = self.fc12(img_feat)

        # Compute attention distribution
        h1 = self.tanh(ques_emb.view(B, 1, self.att_size) + img_emb)
        h1_emb = self.fc13(self.dropout(h1))
        # Mask actual bounding box sizes before calculating softmax
        if v_mask:
            mask = (0 == img_emb.abs().sum(2)).unsqueeze(2).expand(h1_emb.size())
            h1_emb.data.masked_fill_(mask.data, -float('inf'))

        p1 = self.softmax(h1_emb)

        ques_emb_1 = self.fc14(ques_feat)
        img_emb_1 = self.fc15(img_feat)
        #  Compute weighted sum
        img_att_1 = img_emb_1*p1
        weight_sum_1 = torch.sum(img_att_1, dim=1)

        # Combine with question vector
        u1 = ques_emb_1 + weight_sum_1

        # Other stacks
        us = []
        ques_embs = []
        img_embs  = []
        hs = []
        h_embs =[]
        ps  = []
        img_atts = []
        weight_sums = []

        us.append(u1)
        for stack in range(self.num_stacks - 1):
            ques_embs.append(self.layers[3 * stack + 0](us[-1]))
            img_embs.append(self.layers[3 * stack + 1](img_feat))

            # Compute attention distribution
            hs.append(self.tanh(ques_embs[-1].view(B, -1, self.att_size) + img_embs[-1]))
            h_embs.append(self.layers[3*stack + 2](self.dropout(hs[-1])))
            # Mask actual bounding box sizes before calculating softmax
            if v_mask:
                mask = (0 == img_embs[-1].abs().sum(2)).unsqueeze(2).expand(h_embs[-1].size())
                h_embs[-1].data.masked_fill_(mask.data, -float('inf'))
            ps.append(self.softmax(h_embs[-1]))

            #  Compute weighted sum
            img_atts.append(img_embs[-1] * ps[-1])
            weight_sums.append(torch.sum(img_atts[-1], dim=1))

            # Combine with previous stack
            ux = us[-1] + weight_sums[-1]

            # Combine with previous stack by multiple
            us.append(ux)

        return us[-1]


class SanModel(nn.Module):
    def __init__(self, C, ans_size):
        super(SanModel, self).__init__()

        self.v_att = StackedAttention(2, C.HIDDEN_SIZE, C.HIDDEN_SIZE, C.HIDDEN_SIZE, C.HIDDEN_SIZE, 0.1)
        self.classifier = nn.Sequential(
            weight_norm(nn.Linear(C.HIDDEN_SIZE, C.HIDDEN_SIZE*2), dim=None),
            nn.ReLU(),
            nn.Dropout(C.DROPOUT_R, inplace=True),
            weight_norm(nn.Linear(C.HIDDEN_SIZE*2, ans_size), dim=None)
        )

    def forward(self, v, q):
        
        q = q[:,0,:]
        att = self.v_att(v, q)

        logits = self.classifier(att)

        return logits


