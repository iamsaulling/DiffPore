"""
LDM
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from model.Score_model import TimeEmbedding, UNet
from config import parse_args
args = parse_args()


# learnable condition mapper in level 2
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConditionalEmbedding(nn.Module):
    def __init__(self, dim_z, dim_out):
        super().__init__()
        self.dim_z = dim_z

        self.condEmbedding = nn.Sequential(
            nn.Linear(dim_z, dim_out),
            Swish(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, z, t):
        emb = self.condEmbedding(z)
        return emb


class LDM(nn.Module):
    def __init__(self, image_ch=1, ch=128, num_res_blocks=2, res_ch=32,
                 dim_z=1024, feature_ch=3, feature_size=32, dim_t=1024, dim_y=3,
                 T=1000, ch_mult=[1, 2, 3, 4], dropout=0.1,
                 gamma_min: float = -8.0, gamma_max: float = 8.0):
        super().__init__()
        self.image_ch = image_ch
        self.ch = ch
        self.num_res_blocks = num_res_blocks
        self.res_ch = res_ch
        self.dim_z = dim_z
        self.feature_ch = feature_ch
        self.feature_size = feature_size
        self.dim_t = dim_t
        self.dim_y = dim_y
        self.T = T
        self.ch_mult = ch_mult
        self.dropout = dropout
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.time_embedding = torch.nn.DataParallel(TimeEmbedding(T, ch, dim_t))
        self.cond_embedding = torch.nn.DataParallel(ConditionalEmbedding(dim_z, dim_z))
        self.score_model = torch.nn.DataParallel(UNet(feature_ch * 2, ch, ch_mult, num_res_blocks, dropout, dim_z, dim_t))

    def gammat(self, t):
        return torch.tensor(self.gamma_max + (self.gamma_min - self.gamma_max) * t)

    def sigma2(self, gamma):
        return torch.sigmoid(-gamma)

    def alpha(self, gamma):
        return torch.sqrt(1 - self.sigma2(gamma))

    def variance_preserving_map(self, x, gamma, eps):
        a = self.alpha(gamma)
        var = self.sigma2(gamma)
        return a * x + torch.sqrt(var) * eps

    def latent_loss(self, f):
        # KL z1 with N(0,1) prior
        g_1 = self.gammat(1.0).cuda()
        var_1 = self.sigma2(g_1)
        mean1_sqr = (1. - var_1) * torch.square(f)
        loss_klz = 0.5 * torch.sum(mean1_sqr + var_1 - torch.log(var_1) - 1., dim=(1, 2, 3))
        return loss_klz.mean()

    def diffusion_loss(self, f, cond, y):
        # time embedding
        t = torch.rand(size=(f.shape[0],)).cuda()
        t = torch.true_divide(torch.floor(t * self.T) + 1, self.T)
        temb = self.time_embedding(t)

        cemb = self.cond_embedding(cond, t)  # condition embedding

        # sample z_t
        g_t = self.gammat(t).cuda()
        eps_t = torch.randn(size=f.shape).cuda()
        z_t = self.variance_preserving_map(f, g_t[:, None, None, None], eps_t)

        # compute predicted noise
        uncond_prob = np.random.uniform(low=0.0, high=1.0, size=1)
        if uncond_prob < 0.1:
            cemb = None
        eps_hat = self.score_model(z_t, temb, cemb)

        # compute MSE of predicted noise
        loss_diff_mse = torch.sum(torch.square(eps_t - eps_hat), dim=(1, 2, 3))

        # loss for finite depth T, i.e. discrete time
        s = t - torch.true_divide(1., self.T)
        g_s = self.gammat(s)
        rescaled_loss_diff = .5 * self.T * torch.expm1(g_s - g_t) * loss_diff_mse

        return rescaled_loss_diff.mean()

    def forward(self, f, cond):
        # 1. latent loss
        loss_klz = self.latent_loss(f)

        # 2. diffusion loss
        loss_diff = self.diffusion_loss(f, cond)
        return loss_klz, loss_diff

    @torch.no_grad()
    def sample_step(self, i, T, z_t, cond, guidance_weight=1.0):
        eps = torch.randn(size=z_t.shape).cuda()
        t = torch.ones(size=(z_t.shape[0],)).cuda() * (T - i)
        s = torch.ones(size=(z_t.shape[0],)).cuda() * (T - i - 1)
        t = torch.true_divide(t, T).cuda()
        s = torch.true_divide(s, T).cuda()

        g_s = self.gammat(s).cuda()[:, None, None, None]
        g_t = self.gammat(t).cuda()[:, None, None, None]

        cemb = self.cond_embedding(cond, t)
        temb = self.time_embedding(t)
        eps_hat_cond = self.score_model(z_t, temb, cemb)
        eps_hat_uncond = self.score_model(z_t, temb, None)
        eps_hat = (1. + guidance_weight) * eps_hat_cond - guidance_weight * eps_hat_uncond

        a = torch.sigmoid(g_s)
        b = torch.sigmoid(g_t)
        c = -torch.expm1(g_t - g_s)
        sigma_t = torch.sqrt(self.sigma2(g_t))
        z_s = torch.sqrt(torch.true_divide(a,b)) * (z_t - sigma_t * c * eps_hat) + torch.sqrt((1. - a) * c) * eps
        return z_s

    @torch.no_grad()
    def reconstruct(self, t, f, cond, guidance_weight):
        tn = torch.ceil(t * self.T).long()
        t = torch.true_divide(tn, self.T)
        g_t = self.gammat(t)
        eps = torch.randn(size=f.shape).cuda()
        z_0 = self.variance_preserving_map(f, g_t[:, None, None, None], eps)

        # sample step
        for i in torch.arange((self.T - tn.item()), self.T):
            z_0 = self.sample_step(i.cuda(), self.T, z_0, cond, guidance_weight=guidance_weight)
        g_0 = self.gammat(0.0)
        var_0 = self.sigma2(g_0)
        z_0_rescaled = z_0 / torch.sqrt(1. - var_0)
        mean, log_var = torch.chunk(z_0_rescaled, 2, dim=1)
        mean = mean.view(-1, args.dim_z)  # flatten
        log_var = log_var.view(-1, args.dim_z)  # flatten
        return mean, log_var

    @torch.no_grad()
    def generate(self, cond, guidance_weight):
        # generate start noise
        z_0 = torch.randn(size=(cond.size()[0], self.feature_ch * 2, self.feature_size, self.feature_size)).cuda()

        # sample step
        for i in torch.arange(0, self.T):
            z_0 = self.sample_step(i.cuda(), self.T, z_0, cond, guidance_weight=guidance_weight)
        g_0 = self.gammat(0.0)
        var_0 = self.sigma2(g_0)
        z_0_rescaled = z_0 / torch.sqrt(1. - var_0)
        mean, log_var = torch.chunk(z_0_rescaled, 2, dim=1)
        mean = mean.view(-1, args.dim_z)  # flatten
        log_var = log_var.view(-1, args.dim_z)  # flatten
        return mean, log_var



