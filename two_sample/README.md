This code performs kernel learning for a two-sample test via MMD.

- `learn_kernel.py` does the main work in optimizing a kernel for a fixed dataset.
- `eval_kernel.py` tests the kernel learned by `learn_kernel`.
- `mmd.py` has Theano implementations of the MMD and variance estimators, if you just want to use those in your own code.
- `fixed_run.py` does bandwidth selection for an RBF kernel on a given dataset, like the Blobs experiment from the paper.
- `fixed_eval.py` makes a Jupyter notebook analyzing the results of `fixed_run.py`.

### Requirements

The hardest one is [the `feature/bigtest` branch of Shogun](https://github.com/shogun-toolbox/shogun/tree/feature/bigtest), needed for running the two-sample tests (except for the Hotelling test); this isn't actually required for `learn_kernel.py`, but is for evaluations. Make sure you install the modular Python interface (`-DPythonModular=ON` to cmake).

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
