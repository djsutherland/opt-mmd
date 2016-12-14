This does the GAN training.

This code is based on [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow). Hsiao-Yu Tung (sfish0101@gmail.com) did most of the modifications for the paper; feel free to contact her and/or Dougal Sutherland (dougal@gmail.com) with any questions. (Issues here are fine too.)

## Basic Usage:

First download the MNIST files with `data/mnist/fetch.sh`.

- The original MMD generative model: `./run_mmd.sh`.

- The model maximizing the MMD/variance t-statistic: `./run_tmmd.sh`.

- MMD generative model with adversarially optimized kernel function using GAN
loss: `./run_mmd_fm.sh`.

To visualize e.g. the MMD results with tensorboard: `tensorboard --logdir=logs_mmd --port=1234`.
