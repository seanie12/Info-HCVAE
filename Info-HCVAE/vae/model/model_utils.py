import torch
import torch.nn.functional as F
import numpy as np

from math import pi, sqrt, exp


def softargmax(onehot_x, beta=1e4):
    # last dim is the categorical dim, i.e., dim=-1
    categorial_range = torch.arange(onehot_x.size(-1)).to(onehot_x.device).float()
    return torch.sum(F.softmax(onehot_x*beta, dim=-1) * categorial_range, dim=-1).float()


def gaussian_kernel(n=3, sigma=1):
    r = range(-int(n/2), int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]


def sample_gaussian(mu, logvar, num_samples=None):
    if num_samples is None:
        assert len(mu.size()) == 2 and len(
            logvar.size()) == 2  # shape = (batch, dim)
        return mu + torch.randn_like(mu)*torch.exp(0.5 * logvar)
    else:
        assert len(mu.size()) == len(logvar.size())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if len(mu.size()) == 1:
            return mu.unsqueeze(0) + torch.randn((num_samples, mu.size(0)), device=device)*torch.exp(0.5 * logvar.unsqueeze(0))
        elif len(mu.size()) == 2:
            assert mu.size(0) == 1 and logvar.size(0) == 1
            return mu + torch.randn((num_samples, mu.size(1)), device=device)*torch.exp(0.5 * logvar)


def return_mask_lengths(ids):
    mask = torch.sign(ids).float()
    lengths = torch.sum(mask, 1)
    return mask, lengths


def cal_attn(query, memories, mask):
    mask = (1.0 - mask.float()) * -10000.0
    attn_logits = torch.matmul(query, memories.transpose(-1, -2).contiguous())
    attn_logits = attn_logits + mask
    attn_weights = F.softmax(attn_logits, dim=-1)
    attn_outputs = torch.matmul(attn_weights, memories)
    return attn_outputs, attn_logits


def sample_gumbel(shape, device, eps=1e-10):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor

    # ~Gumbel(0,1), shape=(batch, nza, nzadim)
    gumbels = sample_gumbel(logits.size(), logits.device, eps=eps)
    # ~Gumbel(logits,tau), shape=(batch, nza, nzadim)
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim=dim)  # shape=(batch, nza, nzadim)

    if hard:
        # Straight through.
        _, index = y_soft.max(dim, keepdim=True)  # shape = (batch, nza, 1)
        # sampling one-hot categorical variables
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = (y_hard - y_soft).detach() + y_soft
    else:
        # Re-parametrization trick.
        ret = y_soft
    return ret


def gumbel_latent_var_sampling(num_samples, latent_dim, categorical_dim, device):
    """
    Samples from the latent space and return the corresponding
    image space map.
    :param num_samples: (Int) Number of samples
    :param current_device: (Int) Device to run the model
    :return: (Tensor) with shape (num_samples, latent_dim, categorical_dim)
    """
    # [S x D x Q]
    M = num_samples * latent_dim
    np_y = np.zeros((M, categorical_dim), dtype=np.float32)
    np_y[range(M), np.random.choice(categorical_dim, M)] = 1
    np_y = np.reshape(np_y, [num_samples, latent_dim, categorical_dim])
    z_samples = torch.from_numpy(np_y).to(device)
    return z_samples