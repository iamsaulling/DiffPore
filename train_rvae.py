"""
rVAE training
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
from utils import plot_rec_results

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import parse_args
args = parse_args()


def train_rvae(device, rvae, vgg, train_loader, test_loader, pca_loader):
    print('\n RVAE Training Start >>>\n')

    # ------------------------------------------------------------------------------------
    ''' Parameters '''
    train_epoch = args.train_epoch_rvae
    resume_epoch = args.resume_epoch_rvae
    save_freq = args.save_freq
    batch_size = args.batch_size
    dim_y = args.dim_y
    lr = args.lr_rvae
    perceptual_weight = args.perceptual_weight
    kld_weight = args.kld_weight
    adv_weight = args.adv_weight
    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    ''' Paths '''
    output_path = args.output_path + 'RVAE/'

    log_path = output_path + 'log/'
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
    print_network(rvae, 'rvae', log_path)

    ''' Training and Validation History '''
    train_hist = {'total_loss': [],
                  'kld_loss': [],
                  'recon_loss': [],
                  'perceptual_loss': [],
                  'disc_loss': [],
                  'adv_loss': [],
                  'klz_loss': [],
                  'diff_loss': []
                  }
    test_hist = {'total_loss': [],
                 'kld_loss': [],
                 'recon_loss': [],
                 'perceptual_loss': [],
                 'adv_loss': [],
                 'klz_loss': [],
                 'diff_loss': []
                 }

    ''' Optimizer '''
    opt_rvae = torch.optim.AdamW([{'params': rvae.encoder.parameters(), 'initial_lr': lr}] +
                                 [{'params': rvae.decoder.parameters(), 'initial_lr': lr}] +
                                 [{'params': rvae.cond_embedding.parameters(), 'initial_lr': lr}],
                                 lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-4)
    opt_disc = torch.optim.AdamW([{'params': rvae.discriminator.parameters(), 'initial_lr': lr}],
                                 lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-4)

    scheduler_rvae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rvae, train_epoch, 8e-6, last_epoch=resume_epoch-1)
    scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, train_epoch, 8e-6, last_epoch=resume_epoch-1)
    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # resume training
    if path_to_ckpt is not None and resume_epoch > 0:
        save_file = path_to_ckpt + 'ckpt_epoch_{}.pth'.format(resume_epoch)
        checkpoint = torch.load(save_file)
        # load checkpoint
        rvae.load_state_dict(checkpoint['RVAE_state_dict'])
        opt_rvae.load_state_dict(checkpoint['optimizer_rvae_state_dict'])
        opt_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        # load history
        train_hist = np.load(log_path + 'train_hist_{}.npy'.format(resume_epoch), allow_pickle=True).item()
        test_hist = np.load(log_path + 'test_hist_{}.npy'.format(resume_epoch), allow_pickle=True).item()
    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # training start
    start_tmp = timeit.default_timer()
    for epoch in range(resume_epoch, train_epoch):
        rvae.train()

        # some lists to store epoch losses
        epoch_loss = []
        epoch_kld_loss = []
        epoch_recon_loss = []
        epoch_perceptual_loss = []
        epoch_disc_loss = []
        epoch_adv_loss = []

        # ------------------------------------------------------------------------------------
        # iteration of train_loader
        for iter, (train_images, train_labels) in enumerate(train_loader):
            train_images = train_images.type(torch.float).cuda()
            train_groups = train_labels[:, dim_y]
            train_labels = train_labels[:, :dim_y].type(torch.float).cuda()
            mini_batch = train_images.size()[0]

            if args.disc_loss_type == 'wd':
                loss_d, wd, gp = rvae(train_images, train_labels, vgg, True)
            elif args.disc_loss_type == 'hinge':
                loss_d = rvae(train_images, train_labels, vgg, True)

            opt_disc.zero_grad()
            loss_d.backward()
            opt_disc.step()
            scheduler_disc.step()

            if ((iter + 1) % 5) == 0:
                loss_kld, loss_recon, perceptual_loss_set, loss_adv, adpt_weight = rvae(train_images, train_labels, vgg,
                                                                                     False)
                rescale_to_bpd = torch.true_divide(1.,
                                                   torch.numel(train_images[0]) * torch.log(torch.tensor(2.0))).cuda()

                loss_kld = loss_kld * rescale_to_bpd * kld_weight
                loss_recon = loss_recon * rescale_to_bpd

                loss_perceptual = sum(perceptual_loss_set) * perceptual_weight
                p1 = perceptual_loss_set[0]
                p2 = perceptual_loss_set[1]
                p3 = perceptual_loss_set[2]
                p4 = perceptual_loss_set[3]
                p5 = perceptual_loss_set[4]

                loss = loss_kld + loss_recon + loss_perceptual + loss_adv * adv_weight * adpt_weight

                opt_rvae.zero_grad()
                loss.backward()
                opt_rvae.step()
                scheduler_rvae.step()

                # append epoch loss
                epoch_loss.append(loss.item())
                epoch_kld_loss.append(loss_kld.item())
                epoch_recon_loss.append(loss_recon.item())
                epoch_perceptual_loss.append(loss_perceptual.item())
                epoch_disc_loss.append(loss_d.item())
                epoch_adv_loss.append(loss_adv.item())

                mean, log_var = rvae.encode(train_images)
                log_var_mean = torch.mean(log_var)
                log_var_std = torch.std(log_var)

                if args.disc_loss_type == 'wd':
                    log = "Epoch[{:>3d}][{:>3d}/{:>3d}] " \
                          "D Loss:{:.3f}[wd:{:.3f}, gp:{:.3f}] " \
                          "RVAE Loss:{:.3f}[kld:{:.3f}, rec:{:.3f}, " \
                          "percp:{:.3f}[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}], adv:{:.3f}], " \
                          "log var[mean:{:.3f}, std:{:.3f}], " \
                          "Time:{:.0f}". \
                        format(epoch + 1, iter + 1, train_loader.dataset.__len__() // batch_size,
                               loss_d.item(), wd.item(), gp.item(),
                               loss.item(), loss_kld.item(), loss_recon.item(),
                               loss_perceptual.item(), p1.item(), p2.item(), p3.item(), p4.item(), p5.item(),
                               loss_adv.item() * adv_weight,
                               log_var_mean.item(), log_var_std.item(),
                               timeit.default_timer() - start_tmp)

                elif args.disc_loss_type == 'hinge':
                    log = "Epoch[{:>3d}][{:>3d}/{:>3d}] " \
                          "D Loss:{:.3f} " \
                          "RVAE Loss:{:.3f}[kld:{:.3f}, rec:{:.3f}, " \
                          "percp:{:.3f}[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}], adv:{:.3f}], " \
                          "log var[mean:{:.3f}, std:{:.3f}], " \
                          "Time:{:.0f}". \
                        format(epoch + 1, iter + 1, train_loader.dataset.__len__() // batch_size,
                               loss_d.item(),
                               loss.item(), loss_kld.item(), loss_recon.item(),
                               loss_perceptual.item(), p1.item(), p2.item(), p3.item(), p4.item(), p5.item(),
                               loss_adv.item(),
                               log_var_mean.item(), log_var_std.item(),
                               timeit.default_timer() - start_tmp)

                print(log)
                print_log(log, log_path, log_file_name)

        train_hist['total_loss'].append(np.mean(np.array(epoch_loss)))
        train_hist['kld_loss'].append(np.mean(np.array(epoch_kld_loss)))
        train_hist['recon_loss'].append(np.mean(np.array(epoch_recon_loss)))
        train_hist['perceptual_loss'].append(np.mean(np.array(epoch_perceptual_loss)))
        train_hist['disc_loss'].append(np.mean(np.array(epoch_disc_loss)))
        train_hist['adv_loss'].append(np.mean(np.array(epoch_adv_loss)))
        # ------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------
        # model evaluation
        rvae.eval()
        with torch.no_grad():
            # some lists to store epoch losses
            epoch_loss = []
            epoch_kld_loss = []
            epoch_recon_loss = []
            epoch_perceptual_loss = []
            epoch_adv_loss = []

            for iter, (test_images, test_labels) in enumerate(test_loader):
                test_images = test_images.type(torch.float).cuda()
                test_groups = test_labels[:, dim_y]
                test_labels = test_labels[:, :dim_y].type(torch.float).cuda()
                mini_batch = test_images.size()[0]

                loss_kld, loss_recon, perceptual_loss_set, \
                loss_adv, adpt_weight = rvae.validation(test_images, test_labels, vgg)

                loss_kld = loss_kld * rescale_to_bpd * kld_weight
                loss_recon = loss_recon * rescale_to_bpd
                loss_perceptual = sum(perceptual_loss_set) * perceptual_weight
                loss = loss_kld + loss_recon + loss_perceptual + loss_adv * adv_weight * adpt_weight

                if ((iter + 1) % 10) == 0:
                    # append epoch loss
                    epoch_loss.append(loss.item())
                    epoch_kld_loss.append(loss_kld.item())
                    epoch_recon_loss.append(loss_recon.item())
                    epoch_perceptual_loss.append(loss_perceptual.item())
                    epoch_adv_loss.append(loss_adv.item())

            test_hist['total_loss'].append(np.mean(np.array(epoch_loss)))
            test_hist['kld_loss'].append(np.mean(np.array(epoch_kld_loss)))
            test_hist['recon_loss'].append(np.mean(np.array(epoch_recon_loss)))
            test_hist['perceptual_loss'].append(np.mean(np.array(epoch_perceptual_loss)))
            test_hist['adv_loss'].append(np.mean(np.array(epoch_adv_loss)))

            # plot training reconstructions
            x_recon_train = rvae.reconstruct(train_images)
            fig_name = 'Epoch ' + str(epoch + 1) + '_rec_train.png'
            plot_rec_results(train_images, train_labels, x_recon_train, path_to_train_result, fig_name)

            # plot testing reconstructions
            x_recon_test = rvae.reconstruct(test_images)
            fig_name = 'Epoch ' + str(epoch + 1) + '_rec_test.png'
            plot_rec_results(test_images, test_labels, x_recon_test, path_to_train_result, fig_name)

            if ((epoch + 1) % 100) == 0:
                mean_set = []
                z_set = []
                r_mean_set = []
                r_z_set = []
                index_set = []

                group_types = {
                    '40': 1,
                    '41': 2,
                    '24': 3,
                    '35': 4,
                    '27': 5,
                    '29': 6,
                    '4': 7,
                    '5': 8,
                    '6': 9
                }
                type_set = []

                for iter, (pca_images, pca_labels) in enumerate(pca_loader):
                    pca_images = pca_images.type(torch.float).cuda()
                    pca_groups = pca_labels[:, dim_y]
                    pca_labels = pca_labels[:, :dim_y].type(torch.float).cuda()
                    mini_batch = pca_images.size()[0]

                    pca_means, pca_log_vars = rvae.encode(pca_images)
                    pca_zs = rvae.reparameterize(pca_means, pca_log_vars)
                    pca_r_means = rvae.cond_embedding(pca_labels)
                    pca_eps = torch.randn(size=pca_r_means.shape).cuda()
                    pca_r_zs = pca_r_means + pca_eps

                    for p in range(mini_batch):
                        # latent features
                        pca_mean = pca_means[p].cpu().data.numpy().squeeze()
                        pca_z = pca_zs[p].cpu().data.numpy().squeeze()
                        pca_group = int(pca_groups[p].cpu().data.numpy().squeeze())
                        mean_set.append(pca_mean)
                        z_set.append(pca_z)
                        index_set.append(pca_group)
                        type_set.append(group_types[str(pca_group)])

                        # regularization
                        pca_r_mean = pca_r_means[p].cpu().data.numpy().squeeze()
                        pca_r_z = pca_r_zs[p].cpu().data.numpy().squeeze()
                        r_mean_set.append(pca_r_mean)
                        r_z_set.append(pca_r_z)
                        # index_set.append(pca_group)

                mean_set = np.asarray(mean_set)
                z_set = np.asarray(z_set)
                r_mean_set = np.asarray(r_mean_set)
                r_z_set = np.asarray(r_z_set)

                total_mean_set = np.concatenate((mean_set, r_mean_set))
                total_z_set = np.concatenate((z_set, r_z_set))
                index_set = np.asarray(index_set)
                type_set = np.asarray(type_set)
                pca_result_mean = PCA(n_components=2).fit_transform(total_mean_set)
                pca_result_z = PCA(n_components=2).fit_transform(total_z_set)

                plot_reduction(pca_result_z, index_set, type_set,
                               path_to_train_result + 'Epoch ' + str(epoch + 1) + '_pca_z.png')

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
                    'RVAE_state_dict': rvae.state_dict(),
                    'optimizer_rvae_state_dict': opt_rvae.state_dict(),
                    'optimizer_disc_state_dict': opt_disc.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)

            np.save(log_path + 'train_hist_{}.npy'.format(epoch + 1), train_hist)
            np.save(log_path + 'test_hist_{}.npy'.format(epoch + 1), test_hist)
        # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # end for epoch
    return rvae


def show_loss_hist(train_hist, test_hist, output_path):
    plt.figure(figsize=(10, 8))
    x = range(len(train_hist['total_loss']))

    y1 = train_hist['total_loss']
    y2 = train_hist['kld_loss']
    y3 = train_hist['recon_loss']
    y4 = train_hist['perceptual_loss']
    y5 = train_hist['disc_loss']
    y6 = train_hist['adv_loss']

    z1 = test_hist['total_loss']
    z2 = test_hist['kld_loss']
    z3 = test_hist['recon_loss']
    z4 = test_hist['perceptual_loss']
    z6 = test_hist['adv_loss']

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

    plt.clf()

    plt.plot(x, y5, label='Discriminator loss', color='blue')
    plt.plot(x, y6, label='Adversarial loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    # plt.title('')
    plt.tight_layout()
    save_root = output_path + 'adv_loss_history.png'
    plt.savefig(save_root)

    plt.close()


def plot_reduction(vec, index_set, type_set, save_root):
    vec1, vec2 = np.split(vec, 2, axis=0)
    fig, ax = plt.subplots(figsize=(15, 15))
    # fig, ax = plt.subplots()
    cmap = cm.get_cmap('jet', len(np.unique(type_set)))
    scatter = ax.scatter(vec1[:, 0], vec1[:, 1], s=40, c=type_set, marker='<', cmap=cmap)
    clegend1 = ax.legend(*scatter.legend_elements(num=[1, 2, 3, 4, 5, 6, 7, 8, 9]), loc='upper right')

    scatter = ax.scatter(vec2[:, 0], vec2[:, 1], s=40, c=type_set, marker='d', cmap=cmap)
    clegend2 = ax.legend(*scatter.legend_elements(num=[1, 2, 3, 4, 5, 6, 7, 8, 9]), loc='upper left')

    ax.add_artist(clegend1)
    ax.add_artist(clegend2)
    plt.tight_layout()
    plt.savefig(save_root)
    plt.clf()
    plt.close('all')