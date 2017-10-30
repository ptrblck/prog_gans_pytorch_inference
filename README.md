# Progressive Growing of GANs inference in PyTorch with CelebA training snapshot


## Description
This is an inference sample written in [PyTorch](http://pytorch.org/) of the original Theano/Lasagne code.

I recreated the network as described in the paper of [Karras et al.](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of) 
Since some layers seemed to be missing in PyTorch, these were implemented as well. 
The network and the layers can be found in `model.py`.

For the demo, a [100-celeb-hq-1024x1024-ours snapshot](https://drive.google.com/drive/folders/0B4qLcYyJmiz0bWJ5bHdKT0d6UXc) was used, which was made publicly available by the authors.
Since I couldn't find any model converter between Theano/Lasagne and PyTorch, I used a quick and dirty script to transfer the weights between the models (`transfer_weights.py`).

This repo does not provide the code for training the networks.

### Simple inference
To run the demo, simply execute `predict.py`.
You can specify other weights with the `--weights` flag.

Example image:

![Example image](https://raw.githubusercontent.com/ptrblck/prog_gans_pytorch_inference/master/example_small.png)


### Latent space interpolation
To try the latent space interpolation, use `latent_interp.py`.
All output images will be saved in `./interp`.

The following flags are defined:

  * `--weights` - path to pretrained PyTorch state dict
  * `--output` - Directory for storing interpolated images
  * `--batch_size` - batch size for `DataLoader`
  * `--num_workers` - number of workers for `DataLoader`
  * `--nb_latents` - number of frames to generate
  * `--filter` - gaussian filter length for interpolating latent space
  * `--seed` - random seed for numpy and PyTorch
  * `--cuda` - use GPU 

Example interpolation:

![Example interpolation](https://raw.githubusercontent.com/ptrblck/prog_gans_pytorch_inference/master/example_interp.gif)

### Transferring weights
The pretrained lasagne weights can be transferred to a PyTorch state dict using `transfer_weights.py`.

To transfer other snapshots from the paper (other than CelebA), you have to modify the model architecture accordingly and use the corresponding weights.

### Environment
The code was tested on Ubuntu 16.04 with an NVIDIA GTX 1080 using PyTorch v.0.2.0_4.

`transfer_weights.py` needs Theano and Lasagne to load the pretrained weights.

A single forward pass took approx. 0.031 seconds.


## Links

* [Original code (Theano/Lasagne implementation)](https://github.com/tkarras/progressive_growing_of_gans)

* [Paper (research.nvidia.com)](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of)


## License

This code is a modified form of the original code under the [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/legalcode) license with the following copyright notice:

```
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
```

According the Section 3, I hereby identify [Tero Karras et al. and NVIDIA](https://github.com/tkarras) as the original authors of the material.


