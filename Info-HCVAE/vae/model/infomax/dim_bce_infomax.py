import random
import torch
import torch.nn as nn


class DimBceInfoMax(nn.Module):
    """discriminator network.
    Args:
        x_dim (int): input dim, for example m x n x c for [m, n, c]
        z_dim (int): dimension of latent code (typically a number in [10 - 256])
    """

    def __init__(self, x_dim=784, z_dim=64, use_billinear=False):
        super(DimBceInfoMax, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.use_billinear = use_billinear
        if use_billinear:
            self.discriminator = nn.Bilinear(self.x_dim, self.z_dim, 1)
        else:
            self.discriminator = nn.Sequential(
                nn.Linear(self.x_dim + self.z_dim, 512),
                nn.Mish(True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.Mish(True),
                nn.Linear(256, 128),
                nn.Mish(True),
                nn.Linear(128, 1)
            )
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x, z):
        """
        Inputs:
            x : input from train_loader (batch_size x input_size )
            z : latent codes associated with x (batch_size x z_dim)
        """
        # Generate fake data by shifting
        shift = random.randint(1, x.size(0) - 1)
        fake_x = torch.cat([x[-shift:], x[:-shift]], dim=0)
        fake_z = torch.cat([z[-shift:], z[:-shift]], dim=0)

        if not self.use_billinear:
            true_logits = self.discriminator(torch.cat((x, z), dim=-1))
        else:
            true_logits = self.discriminator(x, z)
        true_labels = torch.ones_like(true_logits)

        if not self.use_billinear:
            fake_z_logits = self.discriminator(torch.cat((x, fake_z), dim=-1))
            fake_x_logits = self.discriminator(torch.cat((fake_x, z), dim=-1))
        else:
            fake_z_logits = self.discriminator(x, fake_z)
            fake_x_logits = self.discriminator(fake_x, z)
        fake_logits = torch.cat([fake_z_logits, fake_x_logits], dim=0)
        fake_labels = torch.zeros_like(fake_logits)

        true_loss = self.bce_loss(true_logits, true_labels)
        fake_loss = 0.5 * self.bce_loss(fake_logits, fake_labels)
        loss_info = true_loss + fake_loss
        return loss_info