# -*- coding: utf-8 -*-
"""
This work is based on the Theano/Lasagne implementation of
Progressive Growing of GANs paper from tkarras:
https://github.com/tkarras/progressive_growing_of_gans

Utils
"""

import numpy as np

from scipy import ndimage
from scipy.misc import imsave

import os

import torch
from torch.utils.data import Dataset


def scale_image(image):
    image -= image.min()
    image /= image.max()
    image *= 255
    return image.astype(np.uint8)


def scale_image_paper(image, drange_in, drange_out):
    '''
    Re-implemented according to
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/misc.py
    '''
    scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
    bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
    image = np.clip(image * scale + bias, 0, 255).astype(np.uint8)
    return image


def save_images(images, output_dir, start_idx=0):
    for i, image in enumerate(images):
        image = scale_image_paper(image, [-1, 1], [0, 255])
        image = image.transpose(1, 2, 0) # CWH -> WHC
        image_path = os.path.join(output_dir,
                                  'image{:04d}.png'.format(i+start_idx))
        imsave(image_path, image)


def get_gaussian_latents(nb_latents, filter_latents):
    latents = np.random.randn(nb_latents, 512, 1, 1).astype(np.float32)
    latents = ndimage.gaussian_filter(latents,
                                      [filter_latents, 0, 0, 0],
                                      mode='wrap')
    latents /= np.sqrt(np.mean(latents**2))
    return latents


class LatentDataset(Dataset):
    def __init__(self, nb_latents=1, filter_latents=1):
        latents = get_gaussian_latents(nb_latents, filter_latents)
        self.data = torch.from_numpy(latents)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
