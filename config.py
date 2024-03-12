"""
Argument settings
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    ''' Overall Settings '''
    parser.add_argument('--root_path', type=str, default='', help='root path')
    parser.add_argument('--data_path', type=str, default='data/raw/', help='data path')
    parser.add_argument('--output_path', type=str, default='train results/', help='training output path')
    parser.add_argument('--seed', type=int, default=2023, metavar='S', help='random seed (default: 2023)')

    ''' Dataset Settings '''
    parser.add_argument('--data_width', type=int, default=8, help='length of velocity list, for group index calculation')
    parser.add_argument('--power', type=list, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument('--power_min', type=float, default=150)
    parser.add_argument('--power_max', type=float, default=400)
    parser.add_argument('--velocity', type=list, default=[0.0 / 1400., 100. / 1400., 200. / 1400., 400. / 1400.,
                                                          700. / 1400., 900. / 1400., 1200. / 1400., 1400. / 1400.])
    parser.add_argument('--velocity_min', type=float, default=100)
    parser.add_argument('--velocity_max', type=float, default=1500)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--image_ch', type=int, default=1)
    parser.add_argument('--randomCrop_num', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=1536)
    parser.add_argument('--num_workers', type=int, default=1)

    ''' Model Settings '''
    parser.add_argument('--dim_z', type=int, default=1024,
                        help='dimensionality of latent feature space, dim_z=feature_ch * feature_size * feature_size')
    parser.add_argument('--feature_ch', type=int, default=1)
    parser.add_argument('--feature_size', type=int, default=32)
    parser.add_argument('--dim_t', type=int, default=1024, help='dimensionality of time embeddings')
    parser.add_argument('--dim_y', type=int, default=2, help='dimensionality of input labels')
    parser.add_argument('--ch', type=int, default=128)
    parser.add_argument('--res_ch', type=int, default=32)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--disc_loss_type', type=str, default='wd', help='discriminator loss, wd or hinge')
    parser.add_argument('--disc_condition', type=bool, default=False, help='whether use condition for discriminator')
    parser.add_argument('--ch_mult', type=list, default=[1, 2, 3, 4])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--T', type=int, default=1000, help='number of sample steps')
    parser.add_argument('--gamma_min', type=float, default=-12.0)
    parser.add_argument('--gamma_max', type=float, default=12.0)
    parser.add_argument('--use_random_model', type=bool, default=False, help='whether to use random model to replace vgg19')

    ''' Training Settings '''
    parser.add_argument('--train_epoch_rvae', type=int, default=1000)
    parser.add_argument('--resume_epoch_rvae', type=int, default=1000)
    parser.add_argument('--train_epoch_ldm', type=int, default=1000)
    parser.add_argument('--resume_epoch_ldm', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=200, help='frequency of saving checkpoints')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_rvae', type=float, default=2e-5)
    parser.add_argument('--lr_ldm', type=float, default=4e-6)
    parser.add_argument('--perceptual_vgg_weights', type=dict, default={'conv1_1': 10,
                                                                        'conv2_1': 10,
                                                                        'conv3_1': 10,
                                                                        'conv4_1': 200,
                                                                        'conv5_1': 200,
                                                                        })
    parser.add_argument('--perceptual_weight', type=float, default=1)
    parser.add_argument('--kld_weight', type=float, default=2.5)
    parser.add_argument('--guidance_weight', type=float, default=1)
    parser.add_argument('--adv_weight', type=float, default=1e-3)

    args = parser.parse_args()

    return args