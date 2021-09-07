
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from transformers import BertPreTrainedModel, BertConfig, BertModel

import numpy as np
import math

from model.ban import BanModel
from model.san import SanModel
from model.mlp import MlpModel


# ----------------------------------------------
# ----------- bilinear Model -------------------
# ----------------------------------------------
class QstEncoder(nn.Module):

    def __init__(self):
        super(QstEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        qst_vec = self.bert(input_ids, token_type_ids, attention_mask)
        qst_feature = qst_vec[2]
        qst_feature = qst_feature[-1]
        
        return qst_feature


class Net(nn.Module):
    def __init__(self, C, answer_size):
        super(Net, self).__init__()

        self.qst_encoder = QstEncoder()

        self.img_feat_linear = nn.Linear(
            C.IMG_FEAT_SIZE,
            C.HIDDEN_SIZE
        )

        if C.model == 'ban':
            self.model = BanModel(C, answer_size)
        elif C.model == 'san':
            self.model = SanModel(C, answer_size)
        elif C.model == 'mlp':
            self.model = MlpModel(C, answer_size)

    def forward(self, img_feat, ques_ix):

        # Pre-process Language Feature
        # with torch.no_grad():
        lang_feat = self.qst_encoder(ques_ix)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        pred = self.model(img_feat, lang_feat)

        return pred


