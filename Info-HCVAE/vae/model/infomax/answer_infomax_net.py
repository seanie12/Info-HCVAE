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
            self.ans_infomax = MineInfoMax(
                x_dim=emsize*seq_len, z_dim=nza*nzadim)
            self.qc_infomax = MineInfoMax(
                x_dim=emsize*2, z_dim=nza*nzadim)
        elif infomax_type == "bce":
            self.ans_infomax = DimBceInfoMax(
                x_dim=emsize*seq_len, z_dim=nza*nzadim, linear_bias=False)
            self.qc_infomax = DimBceInfoMax(
                x_dim=emsize*2, z_dim=nza*nzadim, linear_bias=False)

    def summarize_embeddings(self, emb): # emb shape = (N, seq_len, emsize)
        return torch.sigmoid(torch.mean(emb, dim=1))

    def forward(self, q_embs, c_embs, a_embs, za):
        N, _, _ = q_embs.size()
        q_c_summarized_embs = torch.cat((self.summarize_embeddings(q_embs), \
                                        self.summarize_embeddings(c_embs)), dim=-1)
        return self.ans_infomax(a_embs.view(N, -1), za.view(N, -1)) + 0.5*self.qc_infomax(q_c_summarized_embs, za.view(N, -1))

    def denote_is_infomax_net_for_params(self):
        for param in self.parameters():
            setattr(param, "is_infomax_param", True)
