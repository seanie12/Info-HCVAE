import math
import torch
import torch.nn as nn
from .mine_infomax import MineInfoMax
from .dim_bce_infomax import DimBceInfoMax


class LatentDimMutualInfoMax(nn.Module):
    def __init__(self, embedding, max_seq_len, max_question_len, emsize, nzqdim, nza, nzadim, infomax_type="deep"):
        super(LatentDimMutualInfoMax, self).__init__()
        assert infomax_type in ["deep", "bce"]
        self.embedding = embedding
        self.max_seq_len = max_seq_len
        self.max_question_len = max_question_len
        self.emsize = emsize
        self.nzqdim = nzqdim
        self.nza = nza
        self.nzadim = nzadim
        self.infomax_type = infomax_type

        if infomax_type == "deep":
            self.zq_infomax = MineInfoMax(
                x_dim=emsize*(max_seq_len+max_question_len), z_dim=nzqdim)
            self.za_infomax = MineInfoMax(
                x_dim=emsize*(max_seq_len+max_question_len), z_dim=nza*nzadim)
        elif infomax_type == "bce":
            self.zq_infomax = DimBceInfoMax(x_dim=emsize*(max_seq_len+max_question_len), z_dim=nzqdim)
            self.za_infomax = DimBceInfoMax(x_dim=emsize*(max_seq_len+max_question_len), z_dim=nza*nzadim)

    def forward(self, q_ids, c_ids, a_ids, zq, za):
        N, _ = q_ids.size()

        c_emb = self.embedding(c_ids)
        q_emb = self.embedding(q_ids)
        c_a_emb = self.embedding(c_ids, a_ids, None)
        x_zq = torch.cat([q_emb, c_emb], dim=1).view(N, -1)
        x_za = torch.cat([q_emb, c_a_emb], dim=1).view(N, -1)
        return self.zq_infomax(x_zq, zq), self.za_infomax(x_za, za.view(N, -1))

    def denote_is_infomax_net_for_params(self):
        for param in self.parameters():
            setattr(param, "is_infomax_param", True)
