#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Sample code for inference of Progressive Growing of GANs paper
(https://github.com/tkarras/progressive_growing_of_gans)
using a CelebA snapshot
"""

from __future__ import print_function
import argparse

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


parser = argparse.ArgumentParser(description='Inference demo')
parser.add_argument(
    '--weights',
    default='100_celeb_hq_network-snapshot-010403.pth',
    type=str,
    metavar='PATH',
    help='path to PyTorch state dict')
parser.add_argument('--cuda', dest='cuda', action='store_true')

seed = 2809
use_cuda = False

torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

def run(args):
    global use_cuda
    
    print('Loading Generator')
    model = Generator()
    model.load_state_dict(torch.load(args.weights))
    
    # Generate latent vector
    x = torch.randn(1, 512, 1, 1)
    
    if use_cuda:
        model = model.cuda()
        x = x.cuda()
    
    x = Variable(x, volatile=True)
    
    print('Executing forward pass')
    images = model(x)
    
    if use_cuda:
        images = images.cpu()
    
    images_np = images.data.numpy().transpose(0, 2, 3, 1)
    image_np = scale_image(images_np[0, ...])
    
    print('Output')
    plt.figure()
    plt.imshow(image_np)


def main():
    global use_cuda
    args = parser.parse_args()

    if not args.weights:
        print('No PyTorch state dict path privided. Exiting...')
        return
    
    if args.cuda:
        use_cuda = True

    run(args)

if __name__ == '__main__':
    main()