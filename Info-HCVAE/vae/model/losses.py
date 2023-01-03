import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist
from torch.distributions.gumbel import Gumbel
from model.model_utils import softargmax, gumbel_latent_var_sampling
from model.loss_utils import compute_mmd


class VaeGumbelKLLoss(nn.Module):
    def __init__(self, categorical_dim=10):
        super(VaeGumbelKLLoss, self).__init__()
        self.categorical_dim = categorical_dim

    def forward(self, logits, eps=1e-10):
        logits = F.softmax(logits, dim=-1).view(-1, logits.size(1) * logits.size(2))
        log_ratio = torch.log(logits * self.categorical_dim + eps)
        KLD = torch.sum(logits * log_ratio, dim=-1).mean()
        return KLD


class GumbelKLLoss(nn.Module):
    def __init__(self):
        super(GumbelKLLoss, self).__init__()

    def forward(self, loc_q, scale_q, loc_p, scale_p):
        g_q = Gumbel(loc_q, scale_q)
        g_p = Gumbel(loc_p, scale_p)
        return torch_dist.kl.kl_divergence(g_q, g_p)


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


class VaeGaussianKLLoss(nn.Module):
    def __init__(self):
        super(VaeGaussianKLLoss, self).__init__()

    def forward(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD


class GaussianKLLoss(nn.Module):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu_q, logvar_q, mu_p, logvar_p):
        numerator = logvar_q.exp() + torch.pow(mu_q - mu_p, 2)
        fraction = torch.div(numerator, (logvar_p.exp()))
        kl = 0.5 * torch.sum(logvar_p - logvar_q + fraction - 1, dim=1)
        return kl.mean(dim=0)


class GumbelMMDLoss(nn.Module):
    def __init__(self):
        super(GumbelMMDLoss, self).__init__()

    def forward(self, posterior_z, prior_z):
        _, latent_dim, _ = posterior_z.size()

        # do softargmax to make measuring the mean in MMD possible
        prior_z = softargmax(prior_z)
        posterior_z = softargmax(posterior_z)
        return compute_mmd(posterior_z, prior_z, latent_dim)


class ContinuousKernelMMDLoss(nn.Module):
    def __init__(self):
        super(ContinuousKernelMMDLoss, self).__init__()

    def forward(self, posterior_z, prior_z):
        # input shape = (batch, dim)
        _, latent_dim = posterior_z.size()
        return compute_mmd(posterior_z, prior_z, latent_dim)


class GaussianJensenShannonDivLoss(nn.Module):
    def __init__(self):
        super(GaussianJensenShannonDivLoss, self).__init__()
        self.gaussian_kl_loss = GaussianKLLoss()

    def forward(self, mu1, logvar1, mu2, logvar2):
        mean_mu, mean_logvar = (mu1+mu2) / 2, ((logvar1.exp() + logvar2.exp()) / 2).log()

        loss = self.gaussian_kl_loss(mu1, logvar1, mean_mu, mean_logvar)
        loss += self.gaussian_kl_loss(mu2, logvar2, mean_mu, mean_logvar)
     
        return (0.5 * loss)


class CategoricalJensenShannonDivLoss(nn.Module):
    def __init__(self):
        super(CategoricalJensenShannonDivLoss, self).__init__()

    def forward(self, posterior_za_logits, prior_za_logits):
        posterior_za_probs = F.softmax(posterior_za_logits, dim=1)
        prior_za_probs = F.softmax(prior_za_logits, dim=1)

        mean_probs = 0.5 * (posterior_za_probs + prior_za_probs)
        loss = F.kl_div(F.log_softmax(posterior_za_logits,
                         dim=1), mean_probs, reduction="batchmean")
        loss += F.kl_div(F.log_softmax(prior_za_logits, dim=1),
                         mean_probs, reduction="batchmean")

        return (0.5 * loss)