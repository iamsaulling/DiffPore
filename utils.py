"""
Utils
"""
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import cv2
from config import parse_args
args = parse_args()

black_threshold = (100 / 255.0 - 0.5) / 0.5

'''Print training settings'''


def print_network(net, name, log_path):
    ctime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    f = open(log_path + ctime + '_Network_' + name + '.txt', 'w')
    tmp = sys.stdout
    sys.stdout = f
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

    sys.stdout = tmp
    f.close()


def print_config(log_path):
    ctime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    with open(log_path + ctime + '_Parameters' + '.txt', 'w') as pf:
        pf.write('----------Overall Settings----------\n')
        pf.write('data_path: {}\n'.format(args.data_path))
        pf.write('seed: {}\n'.format(args.seed))
        pf.write('\n')

        pf.write('----------Dataset Settings----------\n')
        pf.write('image_size: {}\n'.format(args.image_size))
        pf.write('image_ch: {}\n'.format(args.image_ch))
        pf.write('randomCrop_num: {}\n'.format(args.randomCrop_num))
        pf.write('crop_size: {}\n'.format(args.crop_size))
        pf.write('\n')

        pf.write('----------Model Settings----------\n')
        pf.write('dim_z: {}\n'.format(args.dim_z))
        pf.write('feature_ch: {}\n'.format(args.feature_ch))
        pf.write('feature_size: {}\n'.format(args.feature_size))
        pf.write('dim_t: {}\n'.format(args.dim_t))
        pf.write('dim_y: {}\n'.format(args.dim_y))
        pf.write('ch: {}\n'.format(args.ch))
        pf.write('res_ch: {}\n'.format(args.res_ch))
        pf.write('num_res_blocks: {}\n'.format(args.num_res_blocks))
        pf.write('disc_loss_type: {}\n'.format(args.disc_loss_type))
        pf.write('disc_condition: {}\n'.format(args.disc_condition))
        pf.write('ch_mult: {}\n'.format(args.ch_mult))
        pf.write('dropout: {}\n'.format(args.dropout))
        pf.write('T: {}\n'.format(args.T))
        pf.write('gamma_min: {}\n'.format(args.gamma_min))
        pf.write('gamma_max: {}\n'.format(args.gamma_max))
        pf.write('use_random_model: {}\n'.format(args.use_random_model))
        pf.write('\n')

        pf.write('----------Training Settings----------\n')
        pf.write('train_epoch_rvae: {}\n'.format(args.train_epoch_rvae))
        pf.write('resume_epoch_rvae: {}\n'.format(args.resume_epoch_rvae))
        pf.write('train_epoch_ldm: {}\n'.format(args.train_epoch_ldm))
        pf.write('resume_epoch_ldm: {}\n'.format(args.resume_epoch_ldm))
        pf.write('save_freq: {}\n'.format(args.save_freq))
        pf.write('batch_size: {}\n'.format(args.batch_size))
        pf.write('lr_rvae: {}\n'.format(args.lr_rvae))
        pf.write('lr_ldm: {}\n'.format(args.lr_ldm))
        pf.write("perceptual_vgg_weights: {}\n".format(args.perceptual_vgg_weights))
        pf.write('perceptual_weight: {}\n'.format(args.perceptual_weight))
        pf.write('kld_weight: {}\n'.format(args.kld_weight))
        pf.write('guidance_weight: {}\n'.format(args.guidance_weight))
        pf.write('adv_weight: {}\n'.format(args.adv_weight))
        pf.write('\n')


def print_log(log, log_path, log_file_name):
    with open(log_path + log_file_name, "a") as pf:
        pf.write(log + "\n")


def plot_pred_results(pred_labels, pred_images, output_path, fig_name):
    plt.figure()
    plt.figure(figsize=(16, 12))

    for p in range(48):
        rp = p + 1
        power = pred_labels[p][0].item()
        velocity = pred_labels[p][1].item()

        # show predicted images
        plt.subplot(6, 8, rp)
        pred_image = pred_images[p].view(args.image_size, args.image_size).cpu().data.numpy()
        black = np.sum(pred_image <= black_threshold)
        porosity_rate = black / (args.image_size * args.image_size)
        if porosity_rate == 0:
            pred_image[0][0] = -1
        plt.imshow(pred_image, cmap='gray')
        plt.title('{:.0f}, {:.0f}, {:.2f}%'.format(power * (args.power_max - args.power_min) + args.power_min,
                                                   velocity * (args.velocity_max - args.velocity_min) + args.velocity_min,
                                                   porosity_rate * 100))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path + fig_name)
    plt.clf()
    plt.close('all')


def plot_rec_results(raw_images, raw_labels, recon_images, output_path, fig_name):
    plt.figure()
    plt.figure(figsize=(15, 15))
    position_index = [1, 2, 3, 4, 5, 6, 7, 8,
                      17, 18, 19, 20, 21, 22, 23, 24,
                      33, 34, 35, 36, 37, 38, 39, 40,
                      49, 50, 51, 52, 53, 54, 55, 56]
    sq = 8
    h = sq * sq // 2
    for p in range(h):
        rp = position_index[p]
        power = raw_labels[p][0].item()
        velocity = raw_labels[p][1].item()

        # show real images
        plt.subplot(sq, sq, rp)
        raw_image = raw_images[p].view(args.image_size, args.image_size).cpu().data.numpy()
        black = np.sum(raw_image <= black_threshold)
        porosity_rate = black / (args.image_size * args.image_size)
        if porosity_rate == 0:
            raw_image[0][0] = -1
        plt.imshow(raw_image, cmap='gray')
        plt.title('{:.0f}, {:.0f}, {:.2f}%'.format(power * (args.power_max - args.power_min) + args.power_min,
                                                 velocity * (args.velocity_max - args.velocity_min) + args.velocity_min,
                                                 porosity_rate * 100))
        plt.xticks([])
        plt.yticks([])

        # show reconstructed images
        plt.subplot(sq, sq, rp + sq)
        recon_image = recon_images[p].view(args.image_size, args.image_size).cpu().data.numpy()
        black = np.sum(recon_image <= black_threshold)
        porosity_rate = black / (args.image_size * args.image_size)
        if porosity_rate == 0:
            recon_image[0][0] = -1
        plt.imshow(recon_image, cmap='gray')
        plt.title('{:.2f}%'.format(porosity_rate * 100))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path + fig_name)
    plt.clf()
    plt.close('all')
