import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from TTmodel import QstEncoder, TrilinearTransformer


# ------------------------------
# -------- model ---------------
# ------------------------------

class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.6, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        #pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class TriTransModel(nn.Module):

    def __init__(self, num_layers=2, ff_dim=3072, dropout=0.1):

        super().__init__()
        self.qst_encoder = QstEncoder()
        self.transformer = TrilinearTransformer(num_layers=num_layers, dim=768, num_heads=12, ff_dim=ff_dim, dropout=dropout)

        self.v_linear = nn.Linear(2048, 768)
        self.logit_fc = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.LayerNorm(768 * 2, eps=1e-12),
            nn.Linear(768 * 2, 2)
        )

        self.loss_fct = BCEFocalLoss(gamma=2, alpha=0.6, reduction='sum')

    def forward(self, input_ids, ans_ids, labels=None, img_data=None):
        
        # Make mask
        q_mask = self.make_mask(input_ids.unsqueeze(2))
        a_mask = self.make_mask(ans_ids.unsqueeze(2))
        v_mask = self.make_mask(img_data)

        qst_feature = self.qst_encoder(input_ids)
        ans_feature = self.qst_encoder(ans_ids)
        img_feature = self.v_linear(img_data)

        img_feature, qst_feature, ans_feature = self.transformer(img_feature, qst_feature, ans_feature, v_mask, q_mask, a_mask)

        combined_feature = ans_feature.sum(1)
        logits = self.logit_fc(combined_feature) 
        
        outputs = (logits,)

        if labels is not None:
            
            loss = self.loss_fct(logits.view(-1, 2), labels.view(-1, 2))

            outputs = (loss,) + outputs

        return outputs

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0) #.unsqueeze(1).unsqueeze(2)