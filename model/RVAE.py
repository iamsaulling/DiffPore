"""
rVAE
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from model.VAE import Encoder, Decoder
from model.Discriminator import Discriminator

from config import parse_args
args = parse_args()


# Regularizer
class ConditionalEmbedding(nn.Module):
    def __init__(self, dim_y, dim_out):
        super().__init__()
        self.dim_y = dim_y

        self.condEmbedding = nn.Sequential(
            nn.Linear(dim_y, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(),

            nn.Linear(dim_out, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(),

            nn.Linear(dim_out, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(),

            nn.Linear(dim_out, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(),

            nn.Linear(dim_out, dim_out)
        )

    def forward(self, y):
        y = y.view(-1, self.dim_y) + 1e-8
        cemb = self.condEmbedding(y)
        return cemb


class RVAE(nn.Module):
    def __init__(self, image_ch=1, ch=128, num_res_blocks=2, res_ch=32,
                 dim_z=1024, feature_ch=3, feature_size=32, dim_t=1024, dim_y=2):
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

        # subnetworks
        self.encoder = torch.nn.DataParallel(Encoder(image_ch, ch, num_res_blocks, res_ch, dim_z))
        self.decoder = torch.nn.DataParallel(Decoder(dim_z, image_ch, ch, num_res_blocks, res_ch))
        self.discriminator = torch.nn.DataParallel(Discriminator(image_ch, ch, dim_z))
        self.cond_embedding = torch.nn.DataParallel(ConditionalEmbedding(dim_y, dim_z))

    def encode(self, x):
        mean, log_var = self.encoder(x)
        return mean, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reshape_feature(self, f):
        b, _ = f.shape
        f = f.view(b, self.feature_ch, self.feature_size, self.feature_size)  # reshape latent features
        return f

    def decode(self, f):
        x_recon = self.decoder(f)
        return x_recon

    def vgg_feature_map(self, input, model):
        features = {}
        x = torch.cat([input, input, input], 1)
        if args.use_random_model:
            layers = {'1': 'relu1',
                      '3': 'relu2',
                      '5': 'relu3',
                      '7': 'relu4',
                      '9': 'relu5',
                      '11': 'relu6',
                      '13': 'relu7',
                      '15': 'relu8',
                      }
            for name, layer in model._modules.items():
                if name not in layers:
                    out = layer(x)
                if name in layers:
                    out = layer(out)
                    features[layers[name]] = out
        else:
            layers = {'2': 'conv1_1',
                      '9': 'conv2_1',
                      '16': 'conv3_1',
                      '29': 'conv4_1',
                      '42': 'conv5_1',
                      }
            for name, layer in model._modules.items():
                x = layer(x)
                if name in layers:
                    features[layers[name]] = x
        return features

    def gram_matrix(self, tensor):
        n, d, h, w = tensor.size()
        features = tensor.view(n, d, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (d * h * w)
        return gram

    def discriminator_loss(self, x, f, cemb=None):
        # inputs
        x_recon = self.decode(f).detach()

        if args.disc_loss_type == 'wd':
            # real
            out_real = self.discriminator(x.detach(), cemb)
            d_real = torch.mean(out_real)

            # fake
            out_fake = self.discriminator(x_recon, cemb)
            d_fake = torch.mean(out_fake)

            # gradient penalty
            alpha = torch.rand((x.shape[0], 1, 1, 1)).cuda()
            x_hat = alpha * x.data + (1 - alpha) * x_recon.data
            x_hat.requires_grad = True
            pred_hat = self.discriminator(x_hat, cemb)

            gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat,
                                            grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gp = 10 * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

            # discriminator loss
            loss_d = - d_real + d_fake + gp
            wd = d_real - d_fake
            return loss_d, wd, gp

        elif args.disc_loss_type == 'hinge':
            # real
            out_real = self.discriminator(x.detach(), cemb)
            d_real = torch.mean(F.relu(1. - out_real))

            # fake
            x_recon = self.decode(f).detach()
            out_fake = self.discriminator(x_recon, cemb)
            d_fake = torch.mean(F.relu(1. + out_fake))

            # discriminator loss
            loss_d = (d_real + d_fake) * 0.5
            return loss_d

    def get_last_layer(self):
        return self.decoder.module.conv_out.weight

    def recon_loss(self, x, f, vgg, cemb=None):
        x_recon = self.decode(f)

        # Smooth L1 loss between x and x_recon
        # loss_sl1 = torch.true_divide(nn.SmoothL1Loss(reduction='sum')(x, x_recon), x.shape[0])
        loss_sl1 = torch.tensor(0.0).cuda()

        # perceptual loss between x and x_recon
        x_features = self.vgg_feature_map(x, vgg)
        recon_features = self.vgg_feature_map(x_recon, vgg)

        perceptual_loss_set = []
        for layer in x_features:
            # x feature map
            x_feature = x_features[layer]
            x_gram = self.gram_matrix(x_feature)

            # reconstruction feature map
            rec_feature = recon_features[layer]
            rec_gram = self.gram_matrix(rec_feature)

            if args.use_random_model:
                layer_perceptual_loss = args.perceptual_vgg_weights[layer] * torch.sum((x_gram - rec_gram) ** 2)
            else:
                layer_perceptual_loss = args.perceptual_vgg_weights[layer] * torch.mean((x_gram - rec_gram) ** 2)

            perceptual_loss_set.append(layer_perceptual_loss)

        if args.disc_loss_type == 'wd':
            # adversarial loss from discriminator
            out_fake = self.discriminator(x_recon, cemb)
            g_fake = torch.mean(out_fake)  # adversarial training

            out_real = self.discriminator(x, cemb)
            g_real = torch.mean(out_real)
            loss_adv = g_real - g_fake

        elif args.disc_loss_type == 'hinge':
            # adversarial loss from discriminator
            out_fake = self.discriminator(x_recon, cemb)
            g_fake = torch.mean(out_fake)  # adversarial training
            loss_adv = - g_fake

        # # adaptive weight, not used, the last layer of decoder should be altered if use
        # nll_grads = torch.autograd.grad(sum(perceptual_loss_set), self.get_last_layer(), retain_graph=True)[0]
        # g_grads = torch.autograd.grad(loss_adv, self.get_last_layer(), retain_graph=True)[0]
        # adpt_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        # adpt_weight = torch.clamp(adpt_weight, 0.0, 1e4).detach()
        adpt_weight = torch.tensor(1.0).cuda()

        return loss_sl1, perceptual_loss_set, loss_adv, adpt_weight

    def kld_loss(self, mean, log_var, cemb):
        loss_kld = -0.5 * torch.sum(1 + log_var - (mean - cemb).pow(2) - log_var.exp(), dim=-1)
        return loss_kld.mean()

    def forward(self, x, cond, vgg, train_d):
        mean, log_var = self.encode(x)
        f = self.reparameterize(mean, log_var)  # sample latent feature

        cemb = self.cond_embedding(cond)  # condition embedding

        signal = None
        if args.disc_condition:
            signal = torch.zeros([cond.size()[0], 2, 256, 256]).cuda()
            for i in range(cond.size()[0]):
                for j in range(2):
                    signal[i, j, :, :] = cond[i, j]

        if train_d:
            if args.disc_loss_type == 'wd':
                loss_d, wd, gp = self.discriminator_loss(x, f, cemb=signal)
                return loss_d, wd, gp

            elif args.disc_loss_type == 'hinge':
                loss_d = self.discriminator_loss(x, f, cemb=signal)
                return loss_d
        else:
            # 1. kld loss
            loss_kld = self.kld_loss(mean, log_var, cemb)

            # 2. reconstruction loss
            loss_recon, perceptual_loss_set, loss_adv, adpt_weight = self.recon_loss(x, f, vgg, cemb=signal)

            return loss_kld, loss_recon, perceptual_loss_set, loss_adv, adpt_weight

    @torch.no_grad()
    def validation(self, x, cond, vgg):
        mean, log_var = self.encode(x)
        f = self.reparameterize(mean, log_var)  # sample latent feature

        cemb = self.cond_embedding(cond)  # condition embedding

        signal = None
        if args.disc_condition:
            signal = torch.zeros([cond.size()[0], 2, 256, 256]).cuda()
            for i in range(cond.size()[0]):
                for j in range(2):
                    signal[i, j, :, :] = cond[i, j]

        # 1. kld loss
        loss_kld = self.kld_loss(mean, log_var, cemb)

        # 2. reconstruction loss
        loss_recon, perceptual_loss_set, loss_adv, adpt_weight = self.recon_loss(x, f, vgg, cemb=signal)

        return loss_kld, loss_recon, perceptual_loss_set, loss_adv, adpt_weight

    @torch.no_grad()
    def reconstruct(self, x):
        mean, log_var = self.encode(x)
        f = self.reparameterize(mean, log_var)
        x_recon = self.decode(f)
        return x_recon

    @torch.no_grad()
    def generate(self, cond):
        pred_emb = self.cond_embedding(cond)
        eps = torch.randn(size=pred_emb.shape).cuda()
        pred_f = pred_emb + eps
        x_gen = self.decode(pred_f)
        return x_gen