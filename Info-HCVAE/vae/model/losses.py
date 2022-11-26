import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import sample_gaussian, gumbel_softmax, sample_gumbel

# Define MMD loss
def compute_kernel(x, y, latent_dim, kernel_bandwidth, imq_scales=[0.1, 0.2, 0.5, 1.0, 2.0, 5, 10.0], kernel="rbf"):
    """ Return a kernel of size (batch_x, batch_y) """
    if kernel == "imq":
        Cbase = 2.0 * latent_dim * kernel_bandwidth ** 2
        imq_scales_cuda = torch.tensor(
            imq_scales, dtype=torch.float).cuda()  # shape = (num_scales,)
        # shape = (num_scales, 1, 1)
        Cs = (imq_scales_cuda * Cbase).unsqueeze(1).unsqueeze(2)
        k = (Cs / (Cs + torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-
             1).pow(2).unsqueeze(0))).sum(dim=0)  # shape = (batch_x, batch_y)
        return k
    elif kernel == "rbf":
        C = 2.0 * latent_dim * kernel_bandwidth ** 2
        return torch.exp(-torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1).pow(2) / C)


def compute_mmd(x, y, latent_dim, kernel_bandwidth=1):
    x_size = x.size(0)
    y_size = y.size(0)
    x_kernel = compute_kernel(x, x, latent_dim, kernel_bandwidth)
    y_kernel = compute_kernel(y, y, latent_dim, kernel_bandwidth)
    xy_kernel = compute_kernel(x, y, latent_dim, kernel_bandwidth)
    mmd_z = (x_kernel - x_kernel.diag().diag()).sum() / ((x_size - 1) * x_size)
    mmd_z_prior = (y_kernel - y_kernel.diag().diag()
                   ).sum() / ((y_size - 1) * y_size)
    mmd_cross = xy_kernel.sum() / (x_size*y_size)
    mmd = mmd_z + mmd_z_prior - 2 * mmd_cross
    return mmd


class CategoricalKLLoss(nn.Module):
    def __init__(self):
        super(CategoricalKLLoss, self).__init__()

    def forward(self, P_logits, Q_logits):
        P = F.softmax(P_logits, dim=-1)
        Q = F.softmax(Q_logits, dim=-1)
        log_P = P.log()
        log_Q = Q.log()
        kl = (P * (log_P - log_Q)).sum(dim=-1).sum(dim=-1)
        return kl.mean(dim=0)


class GaussianKLLoss(nn.Module):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu1, logvar1, mu2, logvar2):
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        fraction = torch.div(numerator, (logvar2.exp()))
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1, dim=1)
        return kl.mean(dim=0)


class CategoricalMMDLoss(nn.Module):
    def __init__(self):
        super(CategoricalMMDLoss, self).__init__()

    def forward(self, posterior_za_logits, prior_za_logits):
        # input shape = (batch, num_samples, dim)
        # batch_size = posterior_za_logits.size(0)
        nlatent = posterior_za.size(1)
        latent_dim = posterior_za_logits.size(2)

        posterior_za = gumbel_softmax(posterior_za_logits, hard=False)
        prior_za = gumbel_softmax(prior_za_logits, hard=False)

        # total_mmd = 0
        # num_samples = 10
        # for idx in range(batch_size):
        #     total_mmd += torch.abs(compute_mmd(posterior_za[idx], prior_za[idx], latent_dim))

        #     # Fake sampling with dropout
        #     dropout_posterior_za = sample_gumbel(posterior_za_logits[idx], hard=False, num_samples=num_samples)
        #     dropout_prior_za = sample_gumbel(prior_za_logits[idx], hard=False, num_samples=num_samples)
        #     for j in range(num_samples):
        #         total_mmd += torch.abs(compute_mmd(dropout_posterior_za[j], dropout_prior_za[j], latent_dim))

        # return total_mmd / ((num_samples+1)*batch_size)
        return compute_mmd(posterior_za.view(-1, nlatent*latent_dim), prior_za.view(-1, nlatent*latent_dim), nlatent*latent_dim)


class ContinuousKernelMMDLoss(nn.Module):
    def __init__(self):
        super(ContinuousKernelMMDLoss, self).__init__()

    def forward(self, posterior_z_mu, posterior_z_logvar, prior_z_mu, prior_z_logvar):
        # input shape = (batch, dim)
        # batch_size = posterior_z_mu.size(0)
        latent_dim = posterior_z_mu.size(1)
        # total_mmd = 0
        posterior_z = sample_gaussian(posterior_z_mu, posterior_z_logvar)
        prior_z = sample_gaussian(prior_z_mu, prior_z_logvar)
        # for idx in range(batch_size):
        #     rand_posterior = sample_gaussian(posterior_z_mu[idx], posterior_z_logvar[idx], num_samples=10)
        #     rand_prior = sample_gaussian(prior_z_mu[idx], prior_z_logvar[idx], num_samples=10)
        #     # Apply dropout to mimic the variations in Q(zq | context) & P(zq | context) distribution
        #     total_mmd += compute_mmd(F.dropout(rand_posterior, p=0.2), F.dropout(rand_prior, p=0.2), latent_dim)
        return compute_mmd(posterior_z, prior_z, latent_dim)
