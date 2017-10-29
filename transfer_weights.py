#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This work is based on the Theano/Lasagne implementation of
Progressive Growing of GANs paper from tkarras:
https://github.com/tkarras/progressive_growing_of_gans

Script for weight transfer (lasagne - PyTorch)
"""

from __future__ import print_function
import argparse

import numpy as np

import os
import cPickle

import torch

import theano
import theano.tensor as T
import lasagne

from model import Generator

parser = argparse.ArgumentParser(description='Weight transfer script')
parser.add_argument(
    '--weights',
    default='',
    type=str,
    metavar='PATH',
    help='path to lasagne checkpoint (default: none)')
parser.add_argument(
    '--output',
    type=str,
    default='./output',
    help='Directory for storing PyTorch weight output')


def init_model(model, conv_weights, wscale_weights, nin_weights,
               nin_wscale_weights):
    for feat_layer, conv_w, wscale_w in zip(model.features, conv_weights,
                                            wscale_weights):
        # Get Conv weights and flip them (lasagne default)
        curr_conv_w = np.copy(conv_w.W.get_value()[:, :, ::-1, ::-1])
        feat_layer.conv.weight.data = torch.FloatTensor(curr_conv_w)
        
        # Get WScale weights
        feat_layer.wscale.scale.data = torch.FloatTensor(
            wscale_w.scale.get_value().reshape(1, ))
        feat_layer.wscale.b.data = torch.FloatTensor(wscale_w.b.get_value())

    # Last layer has to be handeled differently, since a NIN layer was used in
    # lasagne (basically 1x1 conv in PyTorch)
    model.output.conv.weight.data = torch.FloatTensor(
        nin_weights.W.get_value().T).unsqueeze_(2).unsqueeze_(3)
    model.output.wscale.scale.data = torch.FloatTensor(
        nin_wscale_weights.scale.get_value().reshape(1, ))
    model.output.wscale.b.data = torch.FloatTensor(
        nin_wscale_weights.b.get_value())


def compare_results(model, G, use_cuda=False):
    from torch.autograd import Variable

    # Create random latent vector
    example_latents = np.random.randn(1, 512).astype(np.float32)

    # Create theano expressions
    latents_var = T.TensorType(
        'float32', [False] * len(example_latents.shape))('latents_var')
    lod = 0.0
    images_expr = G.eval(
        latents_var, min_lod=lod, max_lod=lod, ignore_unsued_inputs=True)
    gen_fn = theano.function(
        [latents_var], images_expr, on_unused_input='ignore')

    # Generate reference image
    images_ref = gen_fn(example_latents[:1])

    # Use same latent vector for our model (we need [1, 512, 1, 1])
    x = torch.from_numpy(example_latents[:, :, np.newaxis, np.newaxis])

    if use_cuda:
        x = x.cuda()
        model = model.cuda()

    x = Variable(x, volatile=True)
    images = model(x)

    if use_cuda:
        images = images.cpu()

    images = images.data.numpy()
    print('Sum of abs error: {}'.format(np.sum(np.abs(images_ref - images))))


def run(args):
    # Get lasagne weights
    lasagne_weights_path = args.weights

    print('Loading lasagne weights')
    with open(lasagne_weights_path, "rb") as f:
        _, _, G = cPickle.load(f)

    # Set output layer
    lasagne_output_layer = G.find_layer('Glod0S')

    # Get all layers up to output layer
    lasagne_layers = lasagne.layers.get_all_layers(lasagne_output_layer)

    # Get weigths for each layer type
    conv_weights = [l for l in lasagne_layers if 'Conv' in str(l)]

    # Skip last wscale layer weights, since these belong to the NIN layer
    wscale_weights = [l for l in lasagne_layers if 'WScale' in str(l)][:-1]

    # Get NIN weights (these should be the two last layers)
    nin_weights = lasagne_layers[-2]
    nin_wscale_weights = lasagne_layers[-1]  # get last wscale layer weight

    print('Initializing PyTorch model')
    model = Generator()
    init_model(model, conv_weights, wscale_weights, nin_weights,
               nin_wscale_weights)

    if args.output:
        _, model_name = os.path.split(args.weights)
        model_name = model_name.replace('.pkl', '.pth')
        output_path = os.path.join(args.output, model_name)
        print('Saving model to {}'.format(output_path))
        torch.save(model.state_dict(), output_path)


def main():
    args = parser.parse_args()

    if not args.weights:
        print('No lasagne checkpoint defined. Exiting...')
        return

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    run(args)


if __name__ == '__main__':
    main()
