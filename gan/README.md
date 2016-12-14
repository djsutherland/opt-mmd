This does the GAN training.

Code mostly by Hsiao-Yu Tung; feel free to contact her at sfish0101@gmail.com
if you have any questions. (Issues / PRs here are fine too.)

## Basic Usage:

First download the MNIST files with `data/mnist/fetch.sh`.

- The original MMD generative model: `./run_mmd.sh`.

- The model maximizing the MMD/variance t-statistic: `./run_tmmd.sh`.

- MMD generative model with adversarially optimized kernel function using GAN
loss: `./run_mmd_fm.sh`.

To visualize e.g. the MMD results with tensorboard: `tensorboard --logdir=logs_mmd --port=1234`.
