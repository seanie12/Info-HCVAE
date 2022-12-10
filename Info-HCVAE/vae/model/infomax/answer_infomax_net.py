import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mine_infomax import MineInfoMax
from .dim_bce_infomax import DimBceInfoMax


class AnswerLatentDimMutualInfoMax(nn.Module):
    def __init__(self, emsize, seq_len, nza, nzadim, infomax_type="deep"):
        super(AnswerLatentDimMutualInfoMax, self).__init__()
        assert infomax_type in ["deep", "bce"]
        self.emsize = emsize
        self.nza = nza
        self.nzadim = nzadim
        self.infomax_type = infomax_type

        linear_span_dim = 256
        linear_z_dim = 384
        self.span_linear = nn.Linear(seq_len, linear_span_dim, bias=False)
        self.za_linear = nn.Linear(nza*nzadim, linear_z_dim, bias=False)
        self.mish = nn.Mish()

        if infomax_type == "deep":
            self.ans_span_infomax = MineInfoMax(
                x_dim=seq_len, z_dim=nza*nzadim)
            self.global_context_answer_infomax = MineInfoMax(
                x_dim=emsize*seq_len, z_dim=nza*nzadim)
        elif infomax_type == "bce":
            self.ans_span_infomax = DimBceInfoMax(
                x_dim=linear_span_dim, z_dim=linear_z_dim)
            self.global_context_answer_infomax = DimBceInfoMax(
                x_dim=emsize*seq_len, z_dim=linear_z_dim)

    def summarize_embeddings(self, emb):  # emb shape = (N, seq_len, emsize)
        return torch.sigmoid(torch.mean(emb, dim=1))

    def forward(self, c_a_embs, a_ids, za):
        N, _, _ = c_a_embs.size()
        a_act = self.mish(self.span_linear(a_ids.float()))
        za_act = self.mish(self.za_linear(za.view(N, -1)))
        return self.ans_span_infomax(a_act, za_act) + \
            0.25*self.global_context_answer_infomax(c_a_embs.view(N, -1), za_act)

    def denote_is_infomax_net_for_params(self):
        for param in self.parameters():
            setattr(param, "is_infomax_param", True)
