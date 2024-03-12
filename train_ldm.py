"""
LDM training
"""
import os
import torch
import torch.nn as nn
import numpy as np
import timeit
import time

from utils import print_config
from utils import print_network
from utils import print_log
from utils import plot_pred_results
from utils import plot_rec_results
import matplotlib.pyplot as plt

from config import parse_args
args = parse_args()


def train_ldm(device, ldm, rvae, train_loader, test_loader):
    print('\n LDM Training Start >>>\n')

    # ------------------------------------------------------------------------------------
    ''' Parameters '''
    train_epoch = args.train_epoch_ldm
    resume_epoch = args.resume_epoch_ldm
    save_freq = args.save_freq
    batch_size = args.batch_size
    dim_y = args.dim_y
    lr = args.lr_ldm
    guidance_weight = args.guidance_weight
    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    ''' Paths '''
    output_path = args.output_path + 'LDM/'

    log_path = output_path + 'logs/'
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    path_to_ckpt = output_path + 'checkpoint/'
    if not os.path.isdir(path_to_ckpt):
        os.makedirs(path_to_ckpt)

    path_to_train_result = output_path + 'train_result/'
    if not os.path.isdir(path_to_train_result):
        os.makedirs(path_to_train_result)
    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    ''' Log Settings '''
    ctime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    log_file_name = ctime + '_Train log' + '.txt'

    ''' Print Parameter Settings '''
    print_config(log_path)

    ''' Print Models '''
    print_network(ldm, 'ldm', log_path)

    ''' Training and Validation History '''
    train_hist = {'total_loss': [],
                  }
    test_hist = {'total_loss': [],
                 }

    ''' Optimizer '''
    opt_ldm = torch.optim.AdamW([{'params': ldm.parameters(), 'initial_lr': lr}],
                                lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-4)
    scheduler_ldm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ldm, train_epoch, 1e-7, last_epoch=-1)
    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # resume training
    if path_to_ckpt is not None and resume_epoch > 0:
        save_file = path_to_ckpt + 'ckpt_epoch_{}.pth'.format(resume_epoch)
        checkpoint = torch.load(save_file)
        # load checkpoint
        ldm.load_state_dict(checkpoint['LDM_state_dict'])
        opt_ldm.load_state_dict(checkpoint['optimizer_ldm_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        # load history
        train_hist = np.load(log_path + 'train_hist_{}.npy'.format(resume_epoch), allow_pickle=True).item()
        test_hist = np.load(log_path + 'test_hist_{}.npy'.format(resume_epoch), allow_pickle=True).item()
    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # training start
    start_tmp = timeit.default_timer()
    # freeze all RVAE params
    for param in rvae.parameters():
        param.requires_grad_(False)
    rvae.eval()

    for epoch in range(resume_epoch, train_epoch):
        ldm.train()

        # some lists to store epoch losses
        epoch_loss = []
        # ------------------------------------------------------------------------------------
        # iteration of train_loader
        for iter, (train_images, train_labels) in enumerate(train_loader):
            train_images = train_images.type(torch.float).cuda()
            train_groups = train_labels[:, dim_y]
            train_labels = train_labels[:, :dim_y].type(torch.float).cuda()
            mini_batch = train_images.size()[0]

            # compute features
            mean, log_var = rvae.encode(train_images)
            mean = rvae.reshape_feature(mean).detach()
            log_var = rvae.reshape_feature(log_var).detach()
            f = torch.cat((mean, log_var), dim=1)
            cemb = rvae.cond_embedding(train_labels)

            loss_klz, loss_diff = ldm(f, cemb)
            rescale_to_bpd = torch.true_divide(1., 2 * args.dim_z * torch.log(torch.tensor(2.0))).cuda()
            loss = loss_klz * rescale_to_bpd + loss_diff * rescale_to_bpd

            opt_ldm.zero_grad()
            loss.backward()
            opt_ldm.step()
            scheduler_ldm.step()

            if ((iter + 1) % 10) == 0:
                # append epoch loss every 10 iterations
                epoch_loss.append(loss.item())

                log = "Epoch[{:>3d}][{:>3d}/{:>3d}] " \
                      "LDM Loss:{:.3f}, " \
                      "Time:{:.0f}". \
                    format(epoch + 1, iter + 1, train_loader.dataset.__len__() // batch_size,
                           loss.item(),
                           timeit.default_timer() - start_tmp)

                print(log)
                print_log(log, log_path, log_file_name)

        train_hist['total_loss'].append(np.mean(np.array(epoch_loss)))
        # ------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------
        # model evaluation
        ldm.eval()
        with torch.no_grad():
            # some lists to store epoch losses
            epoch_loss = []

            pred_labels = torch.tensor([[1.0, 0.0 / 1400.], [1.0, 1. / 14.], [1.0, 2. / 14.], [1.0, 4. / 14.],
                                        [1.0, 7. / 14.], [1.0, 9. / 14.], [1.0, 12. / 14.], [1.0, 14. / 14.],

                                        [0.8, 0.0 / 1400.], [0.8, 1. / 14.], [0.8, 2. / 14.], [0.8, 4. / 14.],
                                        [0.8, 7. / 14.], [0.8, 9. / 14.], [0.8, 12. / 14.], [0.8, 14. / 14.],

                                        [0.6, 0.0 / 1400.], [0.6, 1. / 14.], [0.6, 2. / 14.], [0.6, 4. / 14.],
                                        [0.6, 7. / 14.], [0.6, 9. / 14.], [0.6, 12. / 14.], [0.6, 14. / 14.],

                                        [0.4, 0.0 / 1400.], [0.4, 1. / 14.], [0.4, 2. / 14.], [0.4, 4. / 14.],
                                        [0.4, 7. / 14.], [0.4, 9. / 14.], [0.4, 12. / 14.], [0.4, 14. / 14.],

                                        [0.2, 0.0 / 1400.], [0.2, 1. / 14.], [0.2, 2. / 14.], [0.2, 4. / 14.],
                                        [0.2, 7. / 14.], [0.2, 9. / 14.], [0.2, 12. / 14.], [0.2, 14. / 14.],

                                        [0.0, 0.0 / 1400.], [0.0, 1. / 14.], [0.0, 2. / 14.], [0.0, 4. / 14.],
                                        [0.0, 7. / 14.], [0.0, 9. / 14.], [0.0, 12. / 14.], [0.0, 14. / 14.]],
                                       dtype=torch.float).cuda()

            for iter, (test_images, test_labels) in enumerate(test_loader):
                test_images = test_images.type(torch.float).cuda()
                test_groups = test_labels[:, dim_y]
                test_labels = test_labels[:, :dim_y].type(torch.float).cuda()
                mini_batch = test_images.size()[0]

                # compute features
                mean_test, log_var_test = rvae.encode(test_images)
                mean_test = rvae.reshape_feature(mean_test)
                log_var_test = rvae.reshape_feature(log_var_test)
                f_test = torch.cat((mean_test, log_var_test), dim=1)
                cemb_test = rvae.cond_embedding(test_labels)

                loss_klz, loss_diff = ldm(f_test, cemb_test)
                loss = loss_klz * rescale_to_bpd + loss_diff * rescale_to_bpd

                if ((iter + 1) % 10) == 0:
                    # append epoch loss every 10 iterations
                    epoch_loss.append(loss.item())

            test_hist['total_loss'].append(np.mean(np.array(epoch_loss)))

            if ((epoch + 1) % 200) == 0:
                # plot training reconstructions
                mean_recon_train, log_var_recon_train = \
                    ldm.reconstruct(torch.tensor([0.8]).cuda(), f, cemb, guidance_weight)
                f_recon_train = rvae.reparameterize(mean_recon_train, log_var_recon_train)
                x_recon_train = rvae.decode(f_recon_train)
                fig_name = 'Epoch ' + str(epoch + 1) + '_rec_train.png'
                plot_rec_results(train_images, train_labels, x_recon_train, path_to_train_result, fig_name)

                # plot testing reconstructions
                mean_recon_test, log_var_recon_test = \
                    ldm.reconstruct(torch.tensor([0.8]).cuda(), f_test, cemb_test, guidance_weight)
                f_recon_test = rvae.reparameterize(mean_recon_test, log_var_recon_test)
                x_recon_test = rvae.decode(f_recon_test)
                fig_name = 'Epoch ' + str(epoch + 1) + '_rec_test.png'
                plot_rec_results(test_images, test_labels, x_recon_test, path_to_train_result, fig_name)

                # plot generated samples
                cemb_pred = rvae.cond_embedding(pred_labels)
                mean_pred, log_var_pred = ldm.generate(cemb_pred, guidance_weight)
                f_pred = rvae.reparameterize(mean_pred, log_var_pred)
                x_pred = rvae.decode(f_pred)
                fig_name = 'Epoch ' + str(epoch + 1) + '_pred.png'
                plot_pred_results(pred_labels, x_pred, path_to_train_result, fig_name)
        # ------------------------------------------------------------------------------------
        # plot training history
        show_loss_hist(train_hist, test_hist, output_path)

        # ------------------------------------------------------------------------------------
        # save checkpoint
        if (epoch + 1) % save_freq == 0 or epoch + 1 == train_epoch:
            save_file = path_to_ckpt + 'ckpt_epoch_{}.pth'.format(epoch + 1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'epoch': epoch,
                    'LDM_state_dict': ldm.state_dict(),
                    'optimizer_ldm_state_dict': opt_ldm.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)

            np.save(log_path + 'train_hist_{}.npy'.format(epoch + 1), train_hist)
            np.save(log_path + 'test_hist_{}.npy'.format(epoch + 1), test_hist)
        # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # end for epoch
    return ldm


def show_loss_hist(train_hist, test_hist, output_path):
    plt.figure(figsize=(10, 8))
    x = range(len(train_hist['total_loss']))

    y1 = train_hist['total_loss']
    z1 = test_hist['total_loss']

    plt.plot(x, y1, label='Training', color='darkcyan')
    plt.plot(x, z1, label='Testing', color='firebrick')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc='upper right')
    plt.grid(True)
    # plt.title('')
    plt.tight_layout()
    save_root = output_path + 'total_loss_history.png'
    plt.savefig(save_root)
    plt.close()

