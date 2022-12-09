import math
import torch
import torch.nn as nn
from model.model_utils import return_mask_lengths

_EPS_ = 1e-6

class _EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / (running_mean + _EPS_) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = _EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean


class _InfoMaxModel(nn.Module):
    """discriminator network.
    Args:
        x_dim (int): input dim, for example m x n x c for [m, n, c]
        z_dim (int): dimension of latent code (typically a number in [10 - 256])
    """
    def __init__(self, x_dim=784, z_dim=64, running_mean_weight=0.1, loss_type="mine"):
        super(_InfoMaxModel, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.loss_type = loss_type
        self.running_mean_weight = running_mean_weight
        self.running_mean = 0
        self.discriminator = nn.Sequential(
            nn.Linear(self.x_dim + self.z_dim, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 400),
            nn.ReLU(True),
            nn.Linear(400, 100),
            nn.ReLU(True),
            nn.Linear(100, 1),
        )

    def forward(self, x, z):
        """
        Inputs:
            x : input from train_loader (batch_size x input_size )
            z : latent codes associated with x (batch_size x z_dim)
        """
        x_z_real = torch.cat((x, z), dim=1)
        x_z_fake = torch.cat((x, self._permute_dims(z)), dim=1)
        d_x_z_real = self.discriminator(x_z_real)
        d_x_z_fake = self.discriminator(x_z_fake)

        neg_info_xz = -d_x_z_real.mean()
        if self.loss_type == "fdiv":
            neg_info_xz += torch.exp(d_x_z_fake - 1).mean()
        elif self.loss_type == "mine":
            second_term, self.running_mean = ema_loss(d_x_z_fake, self.running_mean, self.running_mean_weight)
            neg_info_xz += second_term
        elif self.loss_type == "mine_biased":
            neg_info_xz += (torch.logsumexp(d_x_z_fake, 0) - math.log(d_x_z_fake.size(0)))

        return neg_info_xz


    def _permute_dims(self, z):
        """
        function to permute z based on indicies
        """
        B, _ = z.size()
        perm = torch.randperm(B)
        perm_z = z[perm]
        return perm_z


class ContextualizedInfoMax(nn.Module):
    def __init__(self, context_embedding, emsize, nzqdim, nzadim):
        super(ContextualizedInfoMax, self).__init__()
        self.context_embedding = context_embedding
        self.emsize = emsize
        self.nzqdim = nzqdim
        self.nzadim = nzadim

        self.zq_infomax = _InfoMaxModel(x_dim=emsize*2, z_dim=nzqdim)
        self.za_infomax = _InfoMaxModel(x_dim=emsize*2, z_dim=nzadim)


    def forward(self, q_ids, c_ids, a_ids, zq, za):
        c_mask, _ = return_mask_lengths(c_ids)
        q_mask, _ = return_mask_lengths(q_ids)
        c_cls = self.context_embedding(c_ids, c_mask)[:, 0, :]
        q_cls = self.context_embedding(q_ids, q_mask)[:, 0, :]
        c_a_cls = self.context_embedding(c_ids, c_mask, a_ids)[:, 0, :]
        return self.zq_infomax(torch.cat([q_cls, c_cls], dim=-1), zq), \
            self.za_infomax(torch.cat([q_cls, c_a_cls], dim=-1), za)


    def denote_is_infomax_net_for_params(self):
        for param in self.parameters():
            setattr(param, "is_infomax_param", True)