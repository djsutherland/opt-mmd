This code performs kernel learning for a two-sample test via MMD.

- `learn_kernel.py` does the main work in optimizing a kernel for a fixed dataset.
- `eval_kernel.py` tests the kernel learned by `learn_kernel`.
- `mmd.py` has Theano implementations of the MMD and variance estimators, if you just want to use those in your own code.
- `fixed_run.py` does bandwidth selection for an RBF kernel on a given dataset, like the Blobs experiment from the paper.
- `fixed_eval.py` makes a Jupyter notebook analyzing the results of `fixed_run.py`.

Scripts to exactly reproduce the experiments from the paper coming soon.

### Requirements

The hardest one is [the `feature/bigtest` branch of Shogun](https://github.com/shogun-toolbox/shogun/tree/feature/bigtest), needed for running the two-sample tests (except for the Hotelling test); this isn't actually required for `learn_kernel.py`, but is for evaluations. Make sure you install the modular Python interface (`-DPythonModular=ON` to cmake).

We also need a relatively recent version of [Lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html) to run `learn_kernel.py`, along with some Python standards (scipy, pandas, etc.) and a few other packages. You can get everything but shogun with `pip install -r requirements.txt`.
