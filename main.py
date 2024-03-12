"""
Main
"""
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random

from data_loader import generate_data
from data_loader import generate_dataloader

from model import RVAE
from model import LDM
from model import vgg_model
from model import random_model

from train_rvae import train_rvae
from train_ldm import train_ldm

from config import parse_args
args = parse_args()


''' Seed Settings '''
# ---------------------------------------
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = True
np.random.seed(args.seed)
# ---------------------------------------


def main():

    print('---------------------------------------------------')
    print('-----Diffusion-based pore morphology predictor-----')
    print('---------------------------------------------------')

    ''' Device '''
    # -----------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -----------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------
    ''' Data Loader '''
    train_index_list = [0, 1, 2, 3, 5, 7,
                        8, 9, 11, 12, 14, 15,
                        16, 18, 20, 21, 23,
                        25, 26, 27, 28, 30,
                        32, 33, 34, 36, 37, 38, 39,
                        40, 42, 43, 45, 46]  # 70% training
    test_index_list = [4, 6, 10, 13, 17, 19, 22, 24, 29, 31, 35, 41, 44, 47]  # rest of groups for testing
    pca_index_list = [40, 41, 24, 35, 27, 29, 4, 5, 6]  # groups used for pca visualization during PiPCM training

    processed_data_path = 'data/process/'
    if not os.path.isdir(processed_data_path):
        os.makedirs(processed_data_path)
        generate_data()

    train_loader = generate_dataloader(train_index_list, args.batch_size, shuffle_signal=True)
    test_loader = generate_dataloader(test_index_list, args.batch_size, shuffle_signal=True)
    pca_loader = generate_dataloader(pca_index_list, args.batch_size, shuffle_signal=True)

    # -----------------------------------------------------------------------------------------

    ''' Stage I, process-informed perceptual compression module, rVAE '''
    # -----------------------------------------------------------------------------------------
    rvae = RVAE.RVAE(image_ch=args.image_ch, ch=args.ch, num_res_blocks=args.num_res_blocks,
                     res_ch=args.res_ch, dim_z=args.dim_z, feature_ch=args.feature_ch, feature_size=args.feature_size,
                     dim_t=args.dim_t, dim_y=args.dim_y)
    rvae = nn.DataParallel(rvae)
    rvae = rvae.cuda()
    rvae = rvae.module
    # -----------------------------------------------------------------------------------------

    ''' Pre-trained VGG for perceptual loss computation '''
    # -----------------------------------------------------------------------------------------
    if args.use_random_model:
        vgg = random_model.randomModel
    else:
        vgg = vgg_model.vgg
        pretrained_data = torch.load('model/vgg_normalized.pth')
        vgg.load_state_dict(pretrained_data)

    # freeze all VGG params
    for param in vgg.parameters():
        param.requires_grad_(False)
    vgg = nn.DataParallel(vgg)
    vgg = vgg.cuda()
    vgg = vgg.module

    ''' Train RVAE '''
    # -----------------------------------------------------------------------------------------
    rvae = train_rvae(device, rvae, vgg, train_loader, test_loader, pca_loader)
    # -----------------------------------------------------------------------------------------

    ''' Stage II, self-directed latent denoising module, LDM '''
    # -----------------------------------------------------------------------------------------
    ldm = LDM.LDM(image_ch=args.image_ch, ch=args.ch, num_res_blocks=args.num_res_blocks,
                  res_ch=args.res_ch, dim_z=args.dim_z, feature_ch=args.feature_ch, feature_size=args.feature_size,
                  dim_t=args.dim_t, dim_y=args.dim_y, T=args.T, ch_mult=args.ch_mult,
                  dropout=args.dropout, gamma_min=args.gamma_min, gamma_max=args.gamma_max)
    ldm = nn.DataParallel(ldm)
    ldm = ldm.cuda()
    ldm = ldm.module
    ldm = train_ldm(device, ldm, rvae, train_loader, test_loader)
    # -----------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()