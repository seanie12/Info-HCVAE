import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import sample_gaussian, gumbel_softmax, return_mask_lengths

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
        nlatent = posterior_za_logits.size(1)
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


class DIoUAnswerSpanLoss(nn.Module):
    """ Maybe i'll try later - but it's not sure that this will work """
    def __init__(self):
        super(DIoUAnswerSpanLoss, self).__init__()


    def compute_diou(self, start_positions, end_positions, gt_start_positions, gt_end_positions):
        center_dist = (end_positions - start_positions + 1) / 2
        gt_center_dist = (gt_end_positions - gt_start_positions + 1) / 2
        min_start_positions = torch.min(torch.cat((start_positions.unsqueeze(-1), gt_start_positions.unsqueeze(-1)), dim=-1),
                                    dim=-1)[0] # return tuple of torch.min = (min, min_indices)
        max_end_positions = torch.max(torch.cat((end_positions.unsqueeze(-1), gt_end_positions.unsqueeze(-1)), dim=-1),
                                    dim=-1)[0] # return tuple of torch.max = (max, max_indices)
        center_loss = (center_dist - gt_center_dist).pow(2).sum(dim=-1) / (max_end_positions - min_start_positions).pow(2).sum(dim=-1)

        max_start_positions = torch.max(start_positions, gt_start_positions)[0]
        min_end_positions = torch.min(end_positions, gt_end_positions)[0]
        intersection = torch.max(min_end_positions - max_start_positions, 0)[0].sum(dim=-1)
        union = (((end_positions - start_positions) + (gt_end_positions - gt_start_positions)) - intersection).sum(dim=-1)
        iou = intersection / union
        return ((1 - iou) + center_loss).mean()


    def forward(self, c_ids, gt_start_positions, gt_end_positions, start_logits, end_logits):
        # Extract answer mask, start pos, & end pos using start logits & end logits
        c_mask, _ = return_mask_lengths(c_ids)

        mask = torch.matmul(c_mask.unsqueeze(2).float(),
                            c_mask.unsqueeze(1).float()) # shape = (batch, c_len, c_len)
        mask = torch.triu(mask) == 0
        score = (F.log_softmax(start_logits, dim=1).unsqueeze(2)
                 + F.log_softmax(end_logits, dim=1).unsqueeze(1)) # shape = (batch, c_len, c_len)
        score = score.masked_fill(mask, -10000.0)
        score, start_positions = score.max(dim=1) # score's shape = (batch, c_len)
        score, end_positions = score.max(dim=1)
        start_positions = torch.gather(start_positions,
                                       1,
                                       end_positions.view(-1, 1)).squeeze(1)

        return self.compute_diou(start_positions, end_positions, gt_start_positions, gt_end_positions)