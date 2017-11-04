#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Sample code for inference of Progressive Growing of GANs paper
(https://github.com/tkarras/progressive_growing_of_gans)
using a CelebA snapshot
"""

from __future__ import print_function
import argparse
import os

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from model import Generator
from utils import LatentDataset, save_images


interp_types = ['gauss', 'slerp']
use_cuda = False

parser = argparse.ArgumentParser(description='Interpolation demo')
parser.add_argument(
    '--weights',
    default='100_celeb_hq_network-snapshot-010403.pth',
    type=str,
    metavar='PATH',
    help='path to PyTorch state dict')
parser.add_argument(
    '--output',
    type=str,
    default='./interp',
    help='Directory for storing interpolated imaged')
parser.add_argument(
    '--batch_size',
    default=1,
    type=int,
    help='batch size')
parser.add_argument(
    '--num_workers',
    default=1,
    type=int,
    help='number of workers for DataLoader')
parser.add_argument(
    '--type',
    default='gauss',
    choices=interp_types,
    help='interpolation types: ' +
         ' | '.join(interp_types) +
         ' (default: gauss)')
parser.add_argument(
    '--nb_latents',
    default=10,
    type=int,
    help='number of latent vectors to generate')
parser.add_argument(
    '--filter',
    default=2,
    type=int,
    help='gauss filter length for latent vector smoothing (\'gaus\' interp)')
parser.add_argument(
    '--interp',
    default=50,
    type=int,
    help='interpolation length between latents (\'slerp\' inter)')
parser.add_argument(
    '--seed',
    default=187,
    type=int,
    help='Random seed')
parser.add_argument(
    '--cuda',
    dest='cuda',
    action='store_true',
    help='Use GPU for processing')


def run(args):
    global use_cuda
    
    print('Loading Generator')
    model = Generator()
    model.load_state_dict(torch.load(args.weights))
    
    if use_cuda:
        model = model.cuda()
        pin_memory = True
    else:
        pin_memory = False
    
    # Generate latent data
    latent_dataset = LatentDataset(interp_type=args.type,
                                   nb_latents=args.nb_latents,
                                   filter_latents=args.filter,
                                   nb_interp=args.interp)
    latent_loader = DataLoader(latent_dataset,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=False,
                               pin_memory=pin_memory)
    
    print('Processing')
    for i, data in enumerate(latent_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)

        output = model(data)

        if use_cuda:
            output = output.cpu()
    
        images_np = output.data.numpy()
    
        save_images(images_np, args.output, i*args.batch_size)


def main():
    global use_cuda
    args = parser.parse_args()

    if not args.weights:
        print('No PyTorch state dict path provided. Exiting...')
        return

    if args.cuda:
        use_cuda = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    run(args)

if __name__ == '__main__':
    main()