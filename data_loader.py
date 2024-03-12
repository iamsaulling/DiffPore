import os
from torch.utils import data
import random
import numpy as np
import cv2
from config import parse_args
args = parse_args()


class Processor(data.Dataset):
    def __init__(self, images, labels=None, normalize=False):
        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        if labels is not None:
            if len(self.images) != len(self.labels):
                raise Exception('images (' + str(len(self.images)) + ') and labels (' + str(
                    len(self.labels)) + ') do not have the same length!!!')
        self.normalize = normalize

    def __getitem__(self, index):

        image = self.images[index]

        if self.normalize:
            image = image / 255.0
            image = (image - 0.5) / 0.5

        if self.labels is not None:
            label = self.labels[index]
            return (image, label)
        else:
            return image

    def __len__(self):
        return self.n_images


def get_loader(images, labels, batch_size, shuffle_signal):

    labels = labels.astype(float)
    dataset = Processor(images, labels, normalize=True)

    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=shuffle_signal, num_workers=args.num_workers, drop_last=True, pin_memory=True)

    return dataloader


def data_group(label_index_list, images, labels):
    # parameters
    power = args.power
    velocity = args.velocity
    current_index = 0

    for k in label_index_list:
        folder_dir = 'data/process/' + '{}/'.format(k)
        image_list = os.listdir(folder_dir)
        raw_num = len(image_list)

        row = k // args.data_width
        col = k % args.data_width
        current_power = power[row]
        current_velocity = velocity[col]

        for i in range(raw_num):
            image = cv2.imread(folder_dir + '{}.jpg'.format(i), 0)

            images[current_index][:] = image
            labels[current_index][0] = current_power
            labels[current_index][1] = current_velocity
            labels[current_index][2] = k
            current_index += 1

    return images, labels


def generate_data():
    for k in range(48):
        image_list = os.listdir(args.data_path + '{}/'.format(k))
        raw_num = len(image_list)

        save_path = 'data/process/' + '{}/'.format(k)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        current_index = 0

        for i in range(1, raw_num + 1):
            image = cv2.imread(args.data_path + '{}/'.format(k) + '{} ({}).jpg'.format(k, i), 0)
            height = len(image)
            width = len(image[0])

            for cp in range(args.randomCrop_num):
                sty = np.random.randint(0, height - args.crop_size)
                stx = np.random.randint(0, width - args.crop_size)
                crop_image = image[sty: sty + args.crop_size, stx: stx + args.crop_size]
                after_image = cv2.resize(crop_image, (args.image_size, args.image_size), interpolation=cv2.INTER_CUBIC)

                center = (args.image_size - 1) / 2
                for rot in range(4):
                    M = cv2.getRotationMatrix2D((center, center), -90 * rot, 1.0)
                    rot_image = cv2.warpAffine(after_image, M, (args.image_size, args.image_size))
                    flip_image = cv2.flip(rot_image, 0)

                    cv2.imwrite(save_path + '{}.jpg'.format(current_index), rot_image)
                    current_index += 1
                    cv2.imwrite(save_path + '{}.jpg'.format(current_index), flip_image)
                    current_index += 1


def generate_dataloader(label_index_list, batch_size, shuffle_signal):
    # calculate total number of images
    total_num = 0

    for k in label_index_list:
        folder_dir = 'data/process/' + '{}/'.format(k)
        image_list = os.listdir(folder_dir)
        raw_num = len(image_list)
        total_num += raw_num

    images = np.zeros([total_num, args.image_ch, args.image_size, args.image_size])
    labels = np.zeros([total_num, args.dim_y + 1])
    images, labels = data_group(label_index_list, images, labels)

    dataloader = get_loader(images, labels, batch_size, shuffle_signal)

    return dataloader



