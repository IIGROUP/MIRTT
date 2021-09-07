import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from TTmodel import QstEncoder, TrilinearTransformer


# ------------------------------
# -------- model ---------------
# ------------------------------
class BertPredictionHeadTransform(nn.Module):
    def __init__(self):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.transform_act_fn = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform()

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class TTpretrainModel(nn.Module):

    def __init__(self, num_layers=2, ff_dim=3072, dropout=0.1):

        super().__init__()
        self.qst_encoder = QstEncoder()
        self.v_linear = nn.Linear(2048, 768)
        self.transformer = TrilinearTransformer(num_layers=num_layers, dim=768, num_heads=12, ff_dim=ff_dim, dropout=dropout)
        self.cls = BertLMPredictionHead(self.qst_encoder.bert.embeddings.word_embeddings.weight)

        self.loss_fct = CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, ans_ids, img_data, q_mask, q_lm_label_ids, a_mask, a_lm_label_ids):

        v_mask = self.make_mask(img_data)

        qst_feature = self.qst_encoder(input_ids)
        ans_feature = self.qst_encoder(ans_ids)
        img_feature = self.v_linear(img_data)

        img_feature, qst_feature, ans_feature = self.transformer(img_feature, qst_feature, ans_feature, v_mask, q_mask, a_mask)
        q_prediction_scores = self.cls(qst_feature)
        a_prediction_scores = self.cls(ans_feature)

        q_masked_lm_loss = self.loss_fct(
                q_prediction_scores.view(-1, self.qst_encoder.bert.embeddings.word_embeddings.weight.size(0)), 
                q_lm_label_ids.view(-1)
            )
        a_masked_lm_loss = self.loss_fct(
                a_prediction_scores.view(-1, self.qst_encoder.bert.embeddings.word_embeddings.weight.size(0)), 
                a_lm_label_ids.view(-1)
            )
        return (q_masked_lm_loss+a_masked_lm_loss)

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0) #.unsqueeze(1).unsqueeze(2)