import math
import torch
import torch.nn as nn
from .mine_infomax import MineInfoMax
from .dim_bce_infomax import DimBceInfoMax


class LatentDimMutualInfoMax(nn.Module):
    def __init__(self, emsize, max_seq_len, max_question_len, nzqdim, nza, nzadim, infomax_type="deep"):
        super(LatentDimMutualInfoMax, self).__init__()
        assert infomax_type in ["deep", "bce"]
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
                x_dim=emsize*max_seq_len + nzqdim, z_dim=nza*nzadim)
        elif infomax_type == "bce":
            self.zq_infomax = DimBceInfoMax(x_dim=emsize*(max_seq_len+max_question_len), z_dim=nzqdim)
            self.za_infomax = DimBceInfoMax(x_dim=emsize*max_seq_len + nzqdim, z_dim=nza*nzadim, linear_bias=False)

    def forward(self, zq, za, c_f, c_a_f=None, q_f=None):
        N, _, _ = c_f.size()

        if q_f is not None:
            x_zq = torch.cat([q_f, c_f], dim=1).view(N, -1)
        else:
            x_zq = c_f.view(N, -1)

        if c_a_f is not None:
            x_za = torch.cat([zq, c_a_f.view(N, -1)], dim=1)
        else:
            x_za = torch.cat([zq, c_f.view(N, -1)], dim=1)

        return self.zq_infomax(x_zq, zq), self.za_infomax(x_za, za.view(N, -1))

    def denote_is_infomax_net_for_params(self):
        for param in self.parameters():
            setattr(param, "is_infomax_param", True)
