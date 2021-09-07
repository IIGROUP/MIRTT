import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from transformers import BertPreTrainedModel, BertConfig, BertModel

# ------------------------------------------------
# --------- attention sum feature ----------------
# ------------------------------------------------

class AttSum(nn.Module):
    def __init__(self, C):
        super(AttSum, self).__init__()
        
        self.att = nn.Sequential(
            nn.Linear(C.HIDDEN_SIZE, C.HIDDEN_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(C.DROPOUT_R),
            nn.Linear(C.HIDDEN_SIZE, 1)
        )
        self.fc = nn.Linear(C.HIDDEN_SIZE, C.HIDDEN_SIZE)
    
    def forward(self, x):
        att_w = self.att(x)
        att_w = F.softmax(att_w, dim=1)
        x_att = torch.sum(att_w * x, dim=1)
        x_att = self.fc(x_att)

        return x_att

# ----------------------------------------------
# ---------------- MLP -------------------------
# ----------------------------------------------


class MlpModel(nn.Module):
    def __init__(self, C, answer_size):
        super(MlpModel, self).__init__()

        self.att_img = AttSum(C)
        
        self.proj = nn.Sequential(
            GeLU(),      
            nn.Linear(C.HIDDEN_SIZE, answer_size)
        )

    def forward(self, v, q):

        q = q[:,0,:]
        v = self.att_img(v)

        pred = v + q
        pred = self.proj(pred)

        return pred


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))