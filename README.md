# Progressive Growing of GANs inference in PyTorch with CelebA training snapshot


## Description
This is an inference sample written in [PyTorch](http://pytorch.org/) of the original Theano/Lasagne code.

I recreated the network as described in the paper of [Karras et al.](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of) 
Since some layers seemed to be missing in PyTorch, these were implemented as well. 
The network and the layers can be found in `model.py`.

For the demo, a [100-celeb-hq-1024x1024-ours snapshot](https://drive.google.com/drive/folders/0B4qLcYyJmiz0bWJ5bHdKT0d6UXc) was used, which was made publicly available by the authors.
Since I couldn't find any model converter between Theano/Lasagne and PyTorch, I used a quick and dirty script to transfer the weights between the models.

This repo does not provide the code for training the networks.

To run the demo, simply execute `predict.py`.

The code was tested on Ubuntu 16.04 with an NVIDIA GTX 1080 using PyTorch v.0.2.0_4.

Example image:
![Example image](https://raw.githubusercontent.com/ptrblck/prog_gans_pytorch_inference/master/example.png =250x)


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


