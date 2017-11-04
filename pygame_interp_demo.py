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
from torch.utils.data.dataloader import DataLoader

from model import Generator
from utils import LatentDataset, scale_image_paper

import pygame

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
    '--num_workers',
    default=1,
    type=int,
    help='number of workers for DataLoader')
parser.add_argument(
    '--type',
    default='gauss',
    choices=interp_types,
    help='interpolation types: ' + ' | '.join(interp_types) +
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
parser.add_argument('--size', default=256, type=int, help='pygame window size')
parser.add_argument('--seed', default=187, type=int, help='Random seed')
parser.add_argument(
    '--cuda', dest='cuda', action='store_true', help='Use GPU for processing')


def run(args):
    global use_cuda

    # Init PYGame
    pygame.init()
    display = pygame.display.set_mode((args.size, args.size), 0)

    print('Loading Generator')
    model = Generator()
    model.load_state_dict(torch.load(args.weights))

    if use_cuda:
        model = model.cuda()
        pin_memory = True
    else:
        pin_memory = False

        # Generate latent data
    latent_dataset = LatentDataset(
        interp_type=args.type,
        nb_latents=args.nb_latents,
        filter_latents=args.filter,
        nb_interp=args.interp)
    latent_loader = DataLoader(
        latent_dataset,
        batch_size=1,  # Since we want see it 'live'
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

        image = output.data.numpy()[0, ...].transpose(1, 2, 0)
        image = np.rot90(scale_image_paper(image, [-1, 1], [0, 255]))
        snapshot = pygame.surfarray.make_surface(image)
        snapshot = pygame.transform.scale(snapshot, (args.size, args.size))
        display.blit(snapshot, (0, 0))
        pygame.display.flip()


def main():
    global use_cuda
    args = parser.parse_args()

    if not args.weights:
        print('No PyTorch state dict path privided. Exiting...')
        return

    if args.cuda:
        use_cuda = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    run(args)


if __name__ == '__main__':
    main()
