import torch
import torch.nn as nn

class InfoMaxModel(nn.Module):
    """discriminator network.
    Args:
        z_dim (int): dimension of latent code (typically a number in [10 - 256])
        x_dim (int): for example m x n x c for [m, n, c]
    """
    def __init__(self, z_dim=2, x_dim=784):
        super(InfoMaxModel, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
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
        d_x_z_real = self.discriminator(x_z_real).squeeze()
        d_x_z_fake = self.discriminator(x_z_fake).squeeze()
        info_xz = -(d_x_z_real.mean() - (torch.exp(d_x_z_fake - 1).mean()))
        return info_xz

    def _permute_dims(self, z):
        """
        function to permute z based on indicies
        """
        B, _ = z.size()
        perm = torch.randperm(B)
        perm_z = z[perm]
        return perm_z