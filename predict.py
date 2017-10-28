#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Sample code for inference of Progressive Growing of GANs paper
(https://github.com/tkarras/progressive_growing_of_gans)
using a CelebA snapshot
"""

import numpy as np

import torch
from torch.autograd import Variable

from model import Generator

import matplotlib.pyplot as plt


def scale_image(image):
    image -= image.min()
    image /= image.max()
    image *= 255
    return image.astype(np.uint8)


seed = 2809
use_cuda = False
weights_path = '100_celeb_hq_network-snapshot-010403.pth'

torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

model = Generator()
model.load_state_dict(torch.load(weights_path))

# Generate latent vector
x = torch.randn(1, 512, 1, 1)

if use_cuda:
    model = model.cuda()
    x = x.cuda()

x = Variable(x, volatile=True)
images = model(x)

if use_cuda:
    images = images.cpu()

images_np = images.data.numpy().transpose(0, 2, 3, 1)
image_np = scale_image(images_np[0, ...])

plt.figure()
plt.imshow(image_np)
