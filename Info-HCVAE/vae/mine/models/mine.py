import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.autograd import Variable

from torchvision import datasets
from torchvision.transforms import transforms

from mine.models.layers import ConcatLayer, CustomSequential

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import mine.utils as utils

torch.autograd.set_detect_anomaly(True)

EPS = 1e-6

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print("Device:", device)


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean


class Mine(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.1, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method

        if method == 'concat':
            if isinstance(T, nn.Sequential):
                self.T = CustomSequential(ConcatLayer(), *T)
            else:
                self.T = CustomSequential(ConcatLayer(), T)
        else:
            self.T = T

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, iters, batch_size, opt=None):

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        for iter in range(1, iters + 1):
            mu_mi = 0
            for x, y in utils.batch(X, Y, batch_size):
                opt.zero_grad()
                loss = self.forward(x, y)
                loss.backward()
                opt.step()

                mu_mi -= loss.item()
            if iter % (iters // 3) == 0:
                pass
                #print(f"It {iter} - MI: {mu_mi / batch_size}")

        final_mi = self.mi(X, Y)
        print(f"Final MI: {final_mi}")
        return final_mi


class T(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.layers = CustomSequential(ConcatLayer(), nn.Linear(x_dim + z_dim, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 400),
                                       nn.ReLU(),
                                       nn.Linear(400, 1))

    def forward(self, x, z):
        return self.layers(x, z)


class MutualInformationEstimator(pl.LightningModule):
    def __init__(self, x_dim, z_dim, T_hidden_size=400, loss='mine', **kwargs):
        super().__init__()
        self.x_dim = x_dim
        self.T = CustomSequential(ConcatLayer(), nn.Linear(x_dim + z_dim, T_hidden_size), nn.ReLU(),
                                  nn.Linear(T_hidden_size, T_hidden_size), nn.ReLU(), nn.Linear(T_hidden_size, 1))

        if not ('alpha' in kwargs):
            kwargs['alpha'] = 0.1
        self.energy_loss = Mine(self.T, loss=loss, alpha=kwargs['alpha'])

        self.kwargs = kwargs

        # self.train_loader = kwargs.get('train_loader')
        # self.test_loader = kwargs.get('test_loader')

    def forward(self, x, z):
        if self.on_gpu:
            x = x.cuda()
            z = z.cuda()

        return self.energy_loss(x, z)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.kwargs['lr'])

    def training_step(self, batch, batch_idx):

        x, z = batch

        if self.on_gpu:
            x = x.cuda()
            z = z.cuda()

        loss = self.energy_loss(x, z)
        mi = -loss
        tensorboard_logs = {'loss': loss, 'mi': mi}
        tqdm_dict = {'loss_tqdm': loss, 'mi': mi}

        return {
            **tensorboard_logs, 'log': tensorboard_logs, 'progress_bar': tqdm_dict
        }

    # def test_step(self, batch, batch_idx):
    #     x, z = batch
    #     loss = self.energy_loss(x, z)

    #     return {
    #         'test_loss': loss, 'test_mi': -loss
    #     }

    # def test_end(self, outputs):
    #     avg_mi = torch.stack([x['test_mi']
    #                           for x in outputs]).mean().detach().cpu().numpy()
    #     tensorboard_logs = {'test_mi': avg_mi}

    #     self.avg_test_mi = avg_mi
    #     return {'avg_test_mi': avg_mi, 'log': tensorboard_logs}

    # def train_dataloader(self):
    #     if self.train_loader:
    #         return self.train_loader

    #     train_loader = torch.utils.data.DataLoader(
    #         FunctionDataset(self.kwargs['N'], self.x_dim,
    #                         self.kwargs['sigma'], self.kwargs['f']),
    #         batch_size=self.kwargs['batch_size'], shuffle=True)
    #     return train_loader

    # def test_dataloader(self):
    #     if self.test_loader:
    #         return self.train_loader

    #     test_loader = torch.utils.data.DataLoader(
    #         FunctionDataset(self.kwargs['N'], self.x_dim,
    #                         self.kwargs['sigma'], self.kwargs['f']),
    #         batch_size=self.kwargs['batch_size'], shuffle=True)
    #     return test_loader


def build_dist(rho):
    mu = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[1, rho], [rho, 1]])
    dist = MultivariateNormal(mu, cov)
    return dist


def function_experiment():
    N = 3000
    lr = 1e-4
    batch_size = 256
    epochs = 200

    def f1(x): return x
    def f2(x): return x**3
    def f3(x): return torch.sin(x)
    sigmas = torch.linspace(0, 0.9, 10)
    fs = [f1, f2, f3]
    dim = 2

    res = []
    for sigma in sigmas:
        for ix, f in enumerate(fs):
            print(f"Experiment: {ix + 1}, Sigma: {sigma}...")

            kwargs = {
                'N': N,
                'sigma': sigma,
                'f': f,
                'lr': lr,
                'batch_size': batch_size
            }

            model = MutualInformationEstimator(
                dim, dim, loss='mine', **kwargs).to(device)
            trainer = Trainer(max_epochs=epochs,
                              early_stop_callback=False, gpus=1)
            trainer.fit(model)
            trainer.test()

            # Append result
            res.append([ix, sigma, model.avg_test_mi])

    res = np.array(res)
    Z = res[:, -1].reshape((len(sigmas), len(fs))).T
    plt.figure()
    plt.imshow(Z, cmap='Blues')
    plt.show()


# def rho_experiment():
#     dim = 20
#     N = 3000
#     lr = 1e-3
#     epochs = 100
#     batch_size = 128

#     x_dim = dim
#     z_dim = dim

#     steps = 20
#     rhos = np.linspace(-0.99, 0.99, steps)
#     res = []

#     # Rho Experiment
#     for rho in rhos:
#         train_loader = torch.utils.data.DataLoader(
#             MultivariateNormalDataset(N, dim, rho), batch_size=batch_size, shuffle=True)
#         test_loader = torch.utils.data.DataLoader(
#             MultivariateNormalDataset(N, dim, rho), batch_size=batch_size, shuffle=True)

#         true_mi = train_loader.dataset.true_mi

#         kwargs = {
#             'lr': lr,
#             'batch_size': batch_size,
#             'train_loader': train_loader,
#             'test_loader': test_loader,
#             'alpha': 1.0
#         }

#         model = MutualInformationEstimator(
#             dim, dim, loss='mine_biased', **kwargs).to(device)
#         trainer = Trainer(max_epochs=epochs, early_stop_callback=False, gpus=1)
#         trainer.fit(model)
#         trainer.test()

#         print("True_mi {}".format(true_mi))
#         print("MINE {}".format(model.avg_test_mi))
#         res.append((rho, model.avg_test_mi, true_mi))

#     res = np.array(res)
#     plt.figure()
#     plt.plot(res[:, 0], res[:, 1], label='MINE')
#     plt.plot(res[:, 0], res[:, 2], linestyle='--', label='True MI')
#     plt.legend()
#     plt.show()


# if __name__ == '__main__':
    # rho_experiment()
    # function_experiment()
    # gan_experiment()
