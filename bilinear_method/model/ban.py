import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from transformers import BertPreTrainedModel, BertConfig, BertModel

import numpy as np
import math

# ---------------------------------------------------
# BAN
# ---------------------------------------------------
class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if ''!=act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if ''!=act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class FCSTL(nn.Module):
    def __init__(self, dims, act='Tanh', dropout=0):
        super(FCSTL, self).__init__()
        layers = [nn.Dropout(dropout),
                  nn.Linear(dims[-2], dims[-1]),
                  nn.Tanh()]
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)


# ---------------------------------------------------
class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2,.5], k=1):
        super(BCNet, self).__init__()
        
        self.c = 32
        self.k = k
        self.v_dim = v_dim; self.q_dim = q_dim
        self.h_dim = h_dim; self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1]) # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        
        if None == h_out:
            pass
        elif h_out <= self.c:  
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v).transpose(1,2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1,2).unsqueeze(2)
            d_ = torch.matmul(v_, q_) # b x h_dim x v x q
            logits = d_.transpose(1,2).transpose(2,3) # b x v x q x h_dim
            return logits.sum(1).sum(1).unsqueeze(1)

        # broadcast Hadamard product, matrix-matrix production
        # fast computation but memory inefficient
        # epoch 1, time: 157.84
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat # broadcast, b x h_out x v x h_dim
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2,3)) # b x h_out x v x q
            logits = logits + self.h_bias
            return logits # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        # epoch 1, time: 304.87
        else: 
            v_ = self.dropout(self.v_net(v)).transpose(1,2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1,2).unsqueeze(2)
            d_ = torch.matmul(v_, q_) # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1,2).transpose(2,3)) # b x v x q x h_out
            return logits.transpose(2,3).transpose(1,2) # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1,2).unsqueeze(2) # b x d x 1 x v
        q_ = self.q_net(q).transpose(1,2).unsqueeze(3) # b x d x q x 1
        logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_) # b x d x 1 x 1
        logits = logits.squeeze(3).squeeze(2)
        if 1 < self.k:
            logits = logits.unsqueeze(1) # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k # sum-pooling
        return logits


# ----------------------------------------------------------
class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v,q) # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits


#  --------------------------------------------------------
class BanModel(nn.Module):
    def __init__(self, C, answer_size):
        super(BanModel, self).__init__()
        self.glimpse = C.gamma
        self.v_att = BiAttention(C.HIDDEN_SIZE, C.HIDDEN_SIZE, C.h_mm, C.gamma)
        b_net = []
        q_prj = []
        for i in range(C.gamma):
            b_net.append(BCNet(C.HIDDEN_SIZE, C.HIDDEN_SIZE, C.h_mm, None, k=1))
            q_prj.append(FCNet([C.HIDDEN_SIZE, C.HIDDEN_SIZE], '', .2))
        
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.fc = nn.Sequential(
            weight_norm(nn.Linear(C.HIDDEN_SIZE, C.HIDDEN_SIZE*2), dim=None),
            nn.ReLU(),
            nn.Dropout(C.DROPOUT_R, inplace=True),
            weight_norm(nn.Linear(C.HIDDEN_SIZE*2, answer_size), dim=None)
        )

    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        q: [batch, seq_length, seq_dim]

        return: logits, not probs
        """
        q_emb_list = [0] * self.glimpse
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q)  # b x g x v x q
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q, att[:, g, :, :])  # b x l x h
            q = self.q_prj[g](b_emb[g].unsqueeze(1)) + q
            q_emb_list[g] = q

        q_emb = torch.stack(q_emb_list, 1).sum(1)
        logits = self.fc(q_emb.sum(1))
        return logits
