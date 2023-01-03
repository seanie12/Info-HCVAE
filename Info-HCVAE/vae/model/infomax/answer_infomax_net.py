import torch.nn as nn
from .mine_infomax import MineInfoMax
from .dim_bce_infomax import DimBceInfoMax


class AnswerLatentDimMutualInfoMax(nn.Module):
    def __init__(self, seq_len, nza, nzadim, infomax_type="deep"):
        super(AnswerLatentDimMutualInfoMax, self).__init__()
        assert infomax_type in ["deep", "bce"]
        self.nza = nza
        self.nzadim = nzadim
        self.infomax_type = infomax_type

        if infomax_type == "deep":
            self.ans_span_infomax = MineInfoMax(
                x_dim=seq_len, z_dim=nza*nzadim)
        elif infomax_type == "bce":
            self.ans_span_infomax = DimBceInfoMax(
                x_dim=seq_len, z_dim=nza*nzadim)

    def forward(self, a_ids, za):
        N, _ = a_ids.size()
        return self.ans_span_infomax(a_ids.float(), za.view(N, -1))

    def denote_is_infomax_net_for_params(self):
        for param in self.parameters():
            setattr(param, "is_infomax_param", True)
