import torch
import torch.nn.functional as F

def sample_gaussian(mu, logvar, num_samples=None):
    if num_samples is None:
        assert len(mu.size()) == 2 and len(logvar.size()) == 2 # shape = (batch, dim)
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


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-20, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()  # ~Gumbel(0,1), shape=(batch, nza, nzadim)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau), shape=(batch, nza, nzadim)
    y_soft = gumbels.softmax(dim) # shape=(batch, nza, nzadim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1] # shape = (batch, nza, 1)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0) # sampling one-hot categorical variables
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Re-parametrization trick.
        ret = y_soft
    return ret