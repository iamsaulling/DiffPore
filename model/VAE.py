"""
VAE
"""
import torch
import torch.nn as nn
from config import parse_args
args = parse_args()


class ResBlock(nn.Module):
    def __init__(self, in_ch, ch):
        super().__init__()
        self.in_ch = in_ch
        self.ch = ch

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, in_ch, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_ch, ch, num_res_blocks, res_ch, dim_z):
        super().__init__()
        self.in_ch = in_ch
        self.ch = ch
        self.num_res_blocks = num_res_blocks
        self.res_ch = res_ch
        self.dim_z = dim_z

        blocks = [
            nn.Conv2d(in_ch, ch // 2, 4, stride=2, padding=1),  # 128
            nn.ReLU(inplace=True),

            nn.Conv2d(ch // 2, ch, 4, stride=2, padding=1),  # 64
            nn.ReLU(inplace=True),

            nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),  # 32
            nn.ReLU(inplace=True),

            nn.Conv2d(ch * 2, ch * 4, 4, stride=2, padding=1),  # 16
            nn.ReLU(inplace=True),

            nn.Conv2d(ch * 4, ch, 3, padding=1),  # 16
        ]

        for i in range(num_res_blocks):
            blocks.append(ResBlock(ch, res_ch))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

        self.linear_mean = nn.Sequential(
            nn.Linear(ch * 16 * 16, dim_z),
        )

        self.linear_log_var = nn.Sequential(
            nn.Linear(ch * 16 * 16, dim_z),
        )

    def forward(self, x):
        out = self.blocks(x)
        b, c, h, w = out.shape
        out = out.view(b, c * h * w)
        mean = self.linear_mean(out)
        log_var = self.linear_log_var(out)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, dim_z, out_ch, ch, num_res_blocks, res_ch):
        super().__init__()
        self.dim_z = dim_z
        self.out_ch = out_ch
        self.ch = ch
        self.num_res_blocks = num_res_blocks
        self.res_ch = res_ch

        self.linear = nn.Sequential(
            nn.Linear(dim_z, ch * 16 * 16),
        )

        blocks = []
        for i in range(num_res_blocks):
            blocks.append(ResBlock(ch, res_ch))  # 16

        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(ch, ch * 4, 3, padding=1))  # 16

        blocks.extend(
            [
                nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),  # 32
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),  # 64
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1),  # 128
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(ch // 2, out_ch, 4, stride=2, padding=1),  # 256
                nn.Tanh(),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z):
        out = self.linear(z)
        b, _ = out.shape
        out = out.view(b, self.ch, 16, 16)
        out = self.blocks(out)

        return out