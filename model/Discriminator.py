"""
Discriminator
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# wd
class Discriminator(nn.Module):
    def __init__(self, in_ch, ch, dim_z):
        super().__init__()
        self.in_ch = in_ch
        self.ch = ch
        self.dim_z = dim_z

        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, ch // 2, kernel_size=4, stride=2, padding=1),  # 128
            nn.InstanceNorm2d(ch // 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch // 2, ch, kernel_size=4, stride=2, padding=1),  # 64
            nn.InstanceNorm2d(ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1),  # 32
            nn.InstanceNorm2d(ch * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1),  # 16
            nn.InstanceNorm2d(ch * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2, padding=1),  # 8
            nn.InstanceNorm2d(ch * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Linear(ch * 8 * 8 * 8, 1)
        )

    def forward(self, x, cemb):
        x = self.fc(x)
        x = x.view(-1, self.ch * 8 * 8 * 8)
        x = self.linear(x)
        return x



# hinge
# class Discriminator(nn.Module):
#     def __init__(self, in_ch, ch, dim_z):
#         super().__init__()
#         self.in_ch = in_ch
#         self.ch = ch
#         self.dim_z = dim_z
#
#         self.fc = nn.Sequential(
#             spectral_norm(nn.Conv2d(in_ch, ch // 2, kernel_size=4, stride=2, padding=1)),  # 128
#             nn.LeakyReLU(0.2, inplace=True),
#
#             spectral_norm(nn.Conv2d(ch // 2, ch, kernel_size=4, stride=2, padding=1)),  # 64
#             nn.LeakyReLU(0.2, inplace=True),
#
#             spectral_norm(nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1)),  # 32
#             nn.LeakyReLU(0.2, inplace=True),
#
#             spectral_norm(nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1)),  # 16
#             nn.LeakyReLU(0.2, inplace=True),
#
#             spectral_norm(nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2, padding=1)),  # 8
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.linear = nn.Sequential(
#             nn.Linear(ch * 8 * 8 * 8, 1)
#         )
#
#     def forward(self, x, cemb):
#         x = self.fc(x)
#         x = x.view(-1, self.ch * 8 * 8 * 8)
#         x = self.linear(x)
#         return x


