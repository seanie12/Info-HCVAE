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
        self.seq_len = seq_len
        self.nza = nza
        self.nzadim = nzadim
        self.infomax_type = infomax_type

        if infomax_type == "deep":
            self.ans_span_infomax = MineInfoMax(
                x_dim=seq_len, z_dim=nza*nzadim)
            self.global_context_answer_infomax = MineInfoMax(
                x_dim=emsize*seq_len, z_dim=nza*nzadim)
        elif infomax_type == "bce":
            self.ans_span_infomax = DimBceInfoMax(
                x_dim=seq_len, z_dim=nza*nzadim, linear_bias=False)
            self.global_context_answer_infomax = DimBceInfoMax(
                x_dim=emsize*seq_len, z_dim=nza*nzadim, linear_bias=False)

    def summarize_embeddings(self, emb):  # emb shape = (N, seq_len, emsize)
        return torch.sigmoid(torch.mean(emb, dim=1))

    def forward(self, c_a_embs, a_ids, za):
        N, _, _ = c_a_embs.size()
        return self.ans_span_infomax(a_ids.float(), za.view(N, -1)) + \
            0.25*self.global_context_answer_infomax(c_a_embs.view(N, -1), za.view(N, -1))

    def denote_is_infomax_net_for_params(self):
        for param in self.parameters():
            setattr(param, "is_infomax_param", True)
