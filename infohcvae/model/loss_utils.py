import torch


def compute_kernel(x1, x2, z_dim, kernel_type="imq"):
    # Convert the tensors into row and column vectors
    D = x1.size(1)
    N = x1.size(0)

    x1 = x1.unsqueeze(-2)  # Make it into a column tensor
    x2 = x2.unsqueeze(-3)  # Make it into a row tensor

    """
    Usually the below lines are not required, especially in our case,
    but this is useful when x1 and x2 have different sizes
    along the 0th dimension.
    """
    x1 = x1.expand(N, N, D)
    x2 = x2.expand(N, N, D)

    if kernel_type == "rbf":
        result = compute_rbf(x1, x2, z_dim=z_dim)
    elif kernel_type == "imq":
        result = compute_inv_mult_quad(x1, x2, z_dim=z_dim)
    else:
        raise ValueError('Undefined kernel type.')

    return result


def compute_rbf(x1, x2, z_dim, z_var=2.):
    """
    Computes the RBF Kernel between x1 and x2.
    :param x1: (Tensor)
    :param x2: (Tensor)
    :param eps: (Float)
    :return:
    """
    z_dim = x2.size(-1)
    sigma = 2. * z_dim * z_var

    result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
    return result


def compute_inv_mult_quad(x1, x2, z_dim, z_var=2., eps=1e-7):
    """
    Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
    given by
            k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
    :param x1: (Tensor)
    :param x2: (Tensor)
    :param eps: (Float)
    :return:
    """
    z_dim = x2.size(-1)
    C = 2 * z_dim * z_var
    kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

    # Exclude diagonal elements
    result = kernel.sum() - kernel.diag().sum()

    return result


def compute_mmd(posterior_z, prior_z, latent_dim):
    prior_z_kernel = compute_kernel(prior_z, prior_z, z_dim=latent_dim)
    posterior_z_kernel = compute_kernel(
        posterior_z, posterior_z, z_dim=latent_dim)
    combined_kernel = compute_kernel(prior_z, posterior_z, z_dim=latent_dim)

    mmd = prior_z_kernel.mean() + posterior_z_kernel.mean() - \
        2 * combined_kernel.mean()
    return mmd