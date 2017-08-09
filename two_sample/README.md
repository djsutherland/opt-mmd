This code performs kernel learning for a two-sample test via MMD.

- `learn_kernel.py` does the main work in optimizing a kernel for a fixed dataset.
- `eval_kernel.py` tests the kernel learned by `learn_kernel`.
- `mmd.py` has Theano implementations of the MMD and variance estimators, if you just want to use those in your own code.
- `fixed_run.py` does bandwidth selection for an RBF kernel on a given dataset, like the Blobs experiment from the paper.
- `fixed_eval.py` makes a Jupyter notebook analyzing the results of `fixed_run.py`.

### Requirements

The hardest one is [Shogun](http://shogun.ml), version at least 6.0, needed for running the two-sample tests; this isn't actually required for `learn_kernel.py`, but is for evaluations. Make sure you install the modular Python interface (`-DPythonModular=ON` to cmake). The easiest way to get it is [from `conda-forge`](https://github.com/shogun-toolbox/shogun/blob/develop/doc/readme/INSTALL.md#anaconda).

We also need a relatively recent version of [Lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html) to run `learn_kernel.py`, along with some Python standards (scipy, pandas, etc.) and a few other packages. You can get everything but shogun with `pip install -r requirements.txt`.


### Blobs experiment

To reproduce the figures from the Blobs experiment:

- First, create the results files with `fixed_run.py`. This will take a while, since it's doing lots of replications:

```
for r in 1 2 4 6 8 10; do
  python fixed_run.py -n 500 --blobs $r
done
```

- To look at the powers and the bandwidths chosen for a single parameter setting, like Figure 2b of the paper, run e.g. `python fixed_eval.py res_fixed/blobs/rat6/n500.h5` and then look at the Jupyter notebook created in the same folder.

- To plot the rejection rates across different parameter settings, like Figure 2c of the paper, see the file [`fixed_overall.ipynb`](fixed_overall.ipynb).


### GAN model criticism

To replicate the GAN model criticism from the paper, first you'll need samples from a trained GAN.

The [`generate.sample_mnist_minibatch_gan`](generate.py#L136) function handles sampling from an [Improved GAN model](https://github.com/openai/improved-gan). Unfortunately, their repo doesn't currently include code for minibatch discrimination on MNIST. My fork has a file that does what the authors told me by email that they did: [`train_mnist_minibatch_discrimination.py`](https://github.com/dougalsutherland/improved-gan/blob/mnist-minibatch/mnist_svhn_cifar10/train_mnist_minibatch_discrimination.py). Or you can just get [the trained model that I used](https://github.com/dougalsutherland/improved-gan/blob/mnist-minibatch-with-model/mnist_svhn_cifar10/mnist_minibatch_count100_scaled_1.npz?raw=true).

(If you wanted to do this for your own data, you'd want to implement a function in `generate` to sample from it.)

Then, to learn an ARD kernel, I used:
```
THEANO_FLAGS=device=gpu,lib.cnmem=1 python learn_kernel.py \
  --net-version scaling --max-ratio \
  --init-sigma-median --opt-sigma --num-epochs 10000 \
  --{n-train,n-test}=2000 --{,val-}batchsize=500 \
  --mnist-minibatch-gan PATH/TO/mnist_minibatch_count100_scaled_1.npz \
  --trim-edges --scaled --bw \
  results/gan/ard_maxratio_bw_trim.npz
```
(10,000 iterations is overkill; it converges well before that. You might need to change the Theano flags for your device.)

To evaluate the power of the result:
```
python eval_kernel.py --n-reps 100 results/gan/ard_maxratio_bw_trim.npz
```

To evaluate the power of a plain RBF kernel with optimized sigma, do the same thing but with `--net-version nothing`. For the median-heuristic RBF kernel, do `--net-version nothing --no-opt-sigma --num-epochs 0`.
