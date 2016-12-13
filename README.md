Code for the paper "Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy" ([arXiv:1611.04488](https://arxiv.org/abs/1611.04488); [submitted](https://openreview.net/forum?id=HJWHIKqgl) to ICLR 2017).

- General code for learning kernels for a fixed two-sample test, with Theano, is in [two_sample](two_sample); in particular, a Theano implementation of the estimator code is in [`two_sample/mmd.py`](two_sample/mmd.py).
- Code for the GAN models, using TensorFlow, is in [gan](gan). If you just want a TensorFlow implementation of the estimator code, see [`gan/mmd.py`](gan/mmd.py).
- Code for the efficient permutation test described in Section 3 is in [the `feature/bigtest` branch of Shogun](https://github.com/shogun-toolbox/shogun/tree/feature/bigtest); look under [`shogun/src/shogun/statistical_testing`](https://github.com/shogun-toolbox/shogun/tree/feature/bigtest/src/shogun/statistical_testing). We're hoping to get that merged onto mainline shogun soon. An example of using it in the Python API is in [`two_sample/mmd_test.py`](two_sample/mmd_test.py); code to reproduce the experiments coming soon.

This code is under a BSD license, but if you use it, please cite the paper.
