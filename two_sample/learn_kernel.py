from __future__ import division, print_function

import ast
import itertools
import os
import sys
import tempfile
import time
import types

import numpy as np
import lasagne
from six import exec_
from sklearn.metrics.pairwise import euclidean_distances
import theano
import theano.tensor as T

import generate
import mmd


floatX = np.dtype(theano.config.floatX)
def make_floatX(x):
    return np.array(x, dtype=floatX)[()]


################################################################################
################################################################################
### Making the representation network


def net_nothing(net_p, net_q):
    return net_p, net_q, 0


def net_scaling(net_p, net_q):
    net_p = lasagne.layers.ScaleLayer(net_p)
    net_q = lasagne.layers.ScaleLayer(net_q, scales=net_p.scales)
    return net_p, net_q, 0


def net_scaling_exp(net_p, net_q):
    log_scales = theano.shared(np.zeros(net_p.output_shape[1], floatX),
                               name='log_scales')
    net_p = lasagne.layers.ScaleLayer(net_p, scales=T.exp(log_scales))
    net_q = lasagne.layers.ScaleLayer(net_q, scales=net_p.scales)
    return net_p, net_q, 0


def net_rbf(net_p, net_q, J=5):
    '''
    Network equivalent to Wittawat's mean embedding test:
    compute RBF kernel values to each of J test points.
    '''
    from layers import RBFLayer
    net_p = RBFLayer(net_p, J)
    net_q = RBFLayer(net_q, J, locs=net_p.locs, log_sigma=net_p.log_sigma)
    return net_p, net_q, 0


def net_scf(net_p, net_q, n_freqs=5):
    '''
    Network equivalent to Wittawat's smoothed characteristic function test.
    '''
    from layers import SmoothedCFLayer
    net_p = SmoothedCFLayer(net_p, n_freqs)
    net_q = SmoothedCFLayer(net_q, n_freqs,
                            freqs=net_p.freqs, log_sigma=net_p.log_sigma)
    return net_p, net_q, 0


def _paired_dense(in_1, in_2, **kwargs):
    d_1 = lasagne.layers.DenseLayer(in_1, **kwargs)
    d_2 = lasagne.layers.DenseLayer(in_2, W=d_1.W, b=d_1.b, **kwargs)
    return d_1, d_2


def net_basic(net_p, net_q):
    net_p, net_q = _paired_dense(
        net_p, net_q, num_units=128,
        nonlinearity=lasagne.nonlinearities.rectify)
    net_p, net_q = _paired_dense(
        net_p, net_q, num_units=64,
        nonlinearity=lasagne.nonlinearities.rectify)
    return net_p, net_q, 0


net_versions = {
    'nothing': net_nothing,
    'scaling': net_scaling,
    'scaling-exp': net_scaling_exp,
    'rbf': net_rbf,
    'scf': net_scf,
    'basic': net_basic,
}


def register_custom_net(code):
    module = types.ModuleType('net_custom', 'Custom network function')
    exec_(code, module.__dict__)
    sys.modules['net_custom']= module
    net_versions['custom'] = module.net_custom


################################################################################
### Adding loss and so on to the network

def make_network(input_p, input_q, dim,
                 criterion='mmd', biased=True, streaming_est=False,
                 linear_kernel=False, log_sigma=0, hotelling_reg=0,
                 opt_log=True, batchsize=None,
                 net_version='nothing'):

    in_p = lasagne.layers.InputLayer(shape=(batchsize, dim), input_var=input_p)
    in_q = lasagne.layers.InputLayer(shape=(batchsize, dim), input_var=input_q)
    net_p, net_q, reg = net_versions[net_version](in_p, in_q)
    rep_p, rep_q = lasagne.layers.get_output([net_p, net_q])

    choices = {  # criterion, linear kernel, streaming
        ('mmd', False, False): mmd.rbf_mmd2,
        ('mmd', False, True): mmd.rbf_mmd2_streaming,
        ('mmd', True, False): mmd.linear_mmd2,
        ('ratio', False, False): mmd.rbf_mmd2_and_ratio,
        ('ratio', False, True): mmd.rbf_mmd2_streaming_and_ratio,
        ('ratio', True, False): mmd.linear_mmd2_and_ratio,
        ('hotelling', True, False): mmd.linear_mmd2_and_hotelling,
    }
    try:
        fn = choices[criterion, linear_kernel, streaming_est]
    except KeyError:
        raise ValueError("Bad parameter combo: criterion = {}, {}, {}".format(
            criterion,
            "linear kernel" if linear_kernel else "rbf kernel",
            "streaming" if streaming_est else "not streaming"))

    kwargs = {}
    if linear_kernel:
        log_sigma = None
    else:
        log_sigma = theano.shared(make_floatX(log_sigma), name='log_sigma')
        kwargs['sigma'] = T.exp(log_sigma)
    if not streaming_est:
        kwargs['biased'] = biased
    if criterion == 'hotelling':
        kwargs['reg'] = hotelling_reg

    mmd2_pq, stat = fn(rep_p, rep_q, **kwargs)
    obj = -(T.log(T.largest(stat, 1e-6)) if opt_log else stat) + reg
    return mmd2_pq, obj, rep_p, net_p, net_q, log_sigma


################################################################################
### Training helpers

def iterate_minibatches(*arrays, **kwds):
    batchsize = kwds['batchsize']
    shuffle = kwds.get('shuffle', False)

    assert len(arrays) > 0
    n = len(arrays[0])
    assert all(len(a) == n for a in arrays[1:])

    if shuffle:
        indices = np.arange(n)
        np.random.shuffle(indices)

    for start_idx in range(0, max(0, n - batchsize) + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield tuple(a[excerpt] for a in arrays)


def run_train_epoch(X_train, Y_train, batchsize, train_fn):
    total_mmd2 = 0
    total_obj = 0
    n_batches = 0
    batches = itertools.izip( # shuffle the two independently
        iterate_minibatches(X_train, batchsize=batchsize, shuffle=True),
        iterate_minibatches(Y_train, batchsize=batchsize, shuffle=True),
    )
    for ((Xbatch,), (Ybatch,)) in batches:
        mmd2, obj = train_fn(Xbatch, Ybatch)
        assert np.isfinite(mmd2)
        assert np.isfinite(obj)
        total_mmd2 += mmd2
        total_obj += obj
        n_batches += 1
    return total_mmd2 / n_batches, total_obj / n_batches


def run_val(X_val, Y_val, batchsize, val_fn):
    total_mmd2 = 0
    total_obj = 0
    n_batches = 0
    for (Xbatch, Ybatch) in iterate_minibatches(
                X_val, Y_val, batchsize=batchsize):
        mmd2, obj = val_fn(Xbatch, Ybatch)
        assert np.isfinite(mmd2)
        assert np.isfinite(obj)
        total_mmd2 += mmd2
        total_obj += obj
        n_batches += 1
    return total_mmd2 / n_batches, total_obj / n_batches


################################################################################
### Main deal

def setup(dim, criterion='mmd', biased=True, streaming_est=False, opt_log=True,
          linear_kernel=False, opt_sigma=False, init_log_sigma=0,
          net_version='basic', hotelling_reg=0,
          strat='nesterov_momentum', learning_rate=0.01, **opt_args):
    input_p = T.matrix('input_p')
    input_q = T.matrix('input_q')

    mmd2_pq, obj, rep_p, net_p, net_q, log_sigma = make_network(
        input_p, input_q, dim,
        criterion=criterion, biased=biased, streaming_est=streaming_est,
        opt_log=opt_log, linear_kernel=linear_kernel, log_sigma=init_log_sigma,
        hotelling_reg=hotelling_reg, net_version=net_version)

    params = lasagne.layers.get_all_params([net_p, net_q], trainable=True)
    if opt_sigma:
        params.append(log_sigma)
    fn = getattr(lasagne.updates, strat)
    updates = fn(obj, params, learning_rate=learning_rate, **opt_args)

    print("Compiling...", file=sys.stderr, end='')
    train_fn = theano.function(
        [input_p, input_q], [mmd2_pq, obj], updates=updates)
    val_fn = theano.function([input_p, input_q], [mmd2_pq, obj])
    get_rep = theano.function([input_p], rep_p)
    print("done", file=sys.stderr)

    return params, train_fn, val_fn, get_rep, log_sigma


def train(X_train, Y_train, X_val, Y_val,
          criterion='mmd', biased=True, streaming_est=False, opt_log=True,
          linear_kernel=False, hotelling_reg=0,
          init_log_sigma=0, opt_sigma=False, init_sigma_median=False,
          num_epochs=10000, batchsize=200, val_batchsize=1000,
          verbose=True, net_version='basic',
          opt_strat='nesterov_momentum', learning_rate=0.01,
          log_params=False, **opt_args):
    assert X_train.ndim == X_val.ndim == Y_train.ndim == Y_val.ndim == 2
    dim = X_train.shape[1]
    assert X_val.shape[1] == Y_train.shape[1] == Y_val.shape[1] == dim

    if linear_kernel:
        print("Using linear kernel")
    elif opt_sigma:
        print("Starting with sigma = {}; optimizing it".format(
            'median' if init_sigma_median else np.exp(init_log_sigma)))
    else:
        print("Using sigma = {}".format(
            'median' if init_sigma_median else np.exp(init_log_sigma)))

    params, train_fn, val_fn, get_rep, log_sigma = setup(
            dim, criterion=criterion, linear_kernel=linear_kernel,
            biased=biased, streaming_est=streaming_est,
            hotelling_reg=hotelling_reg,
            init_log_sigma=init_log_sigma, opt_sigma=opt_sigma,
            opt_log=opt_log, net_version=net_version,
            strat=opt_strat, learning_rate=learning_rate, **opt_args)

    if log_sigma is not None and init_sigma_median:
        print("Getting median initial sigma value...", end='')
        n_samp = min(500, X_train.shape[0], Y_train.shape[0])
        samp = np.vstack([
            X_train[np.random.choice(X_train.shape[0], n_samp, replace=False)],
            Y_train[np.random.choice(Y_train.shape[0], n_samp, replace=False)],
        ])
        reps = np.vstack([
            get_rep(batch) for batch, in
            iterate_minibatches(samp, batchsize=val_batchsize)])
        D2 = euclidean_distances(reps, squared=True)
        med_sqdist = np.median(D2[np.triu_indices_from(D2, k=1)])
        log_sigma.set_value(make_floatX(np.log(med_sqdist / np.sqrt(2)) / 2))
        rep_dim = reps.shape[1]
        del samp, reps, D2, med_sqdist
        print("{:.3g}".format(np.exp(log_sigma.get_value())))
    else:
        rep_dim = get_rep(X_train[:1]).shape[1]

    print("Input dim {}, representation dim {}".format(
        X_train.shape[1], rep_dim))
    print("Training on {} samples (batch {}), validation on {} (batch {})"
        .format(X_train.shape[0], batchsize, X_val.shape[0], val_batchsize))
    print("{} parameters to optimize: {}".format(
        len(params), ', '.join(p.name for p in params)))

    value_log = np.zeros(num_epochs + 1, dtype=[
            ('train_mmd', floatX), ('train_obj', floatX),
            ('val_mmd', floatX), ('val_obj', floatX),
            ('elapsed_time', np.float64)]
            + ([('sigma', floatX)] if opt_sigma else [])
            + ([('params', object)] if log_params else []))

    fmt = ("{: >6,}: avg train MMD^2 {: .6f} obj {: .6f},  "
           "avg val MMD^2 {: .6f}  obj {: .6f}  elapsed: {:,}s")
    if opt_sigma:
        fmt += '  sigma: {sigma:.3g}'
    def log(epoch, t_mmd2, t_obj, v_mmd2, v_job, t):
        sigma = np.exp(float(params[-1].get_value())) if opt_sigma else None
        if verbose and (epoch in {0, 5, 25, 50}
                # or (epoch < 1000 and epoch % 50 == 0)
                or epoch % 100 == 0):
            print(fmt.format(
                epoch, t_mmd2, t_obj, v_mmd2, v_obj, int(t), sigma=sigma))
        tup = (t_mmd2, t_obj, v_mmd2, v_obj, t)
        if opt_sigma:
            tup += (sigma,)
        if log_params:
            tup += ([p.get_value() for p in params],)
        value_log[epoch] = tup

    t_mmd2, t_obj = run_val(X_train, Y_train, batchsize, val_fn)
    v_mmd2, v_obj = run_val(X_val, Y_val, val_batchsize, val_fn)
    log(0, t_mmd2, t_obj, v_mmd2, v_obj, 0)
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        try:
            t_mmd2, t_obj = run_train_epoch(
                X_train, Y_train, batchsize, train_fn)
            v_mmd2, v_obj = run_val(X_val, Y_val, val_batchsize, val_fn)
            log(epoch, t_mmd2, t_obj, v_mmd2, v_obj, time.time() - start_time)
        except KeyboardInterrupt:
            break

    sigma = np.exp(log_sigma.get_value()) if log_sigma is not None else None
    return ([p.get_value() for p in params], [p.name for p in params],
            get_rep, value_log, sigma)


def eval_rep(get_rep, X, Y, linear_kernel=False, hotelling=False,
             sigma=None, null_samples=1000):
    import mmd_test
    Xrep = get_rep(X)
    Yrep = get_rep(Y)
    if linear_kernel:
        if hotelling:
            p_val, stat, = mmd_test.linear_hotelling_test(Xrep, Yrep)
            null_samps = np.empty(0, dtype=np.float32)
        else:
            p_val, stat, null_samps = mmd_test.linear_mmd_test(
                Xrep, Yrep, null_samples=null_samples)
    else:
        p_val, stat, null_samps, _ = mmd_test.rbf_mmd_test(
            Xrep, Yrep, bandwidth=sigma, null_samples=null_samples)
    return p_val, stat, null_samps


def main():
    import argparse
    parser = argparse.ArgumentParser(description='''
        Learn a kernel function to maximize the power of a two-sample test.
        '''.strip())
    net = parser.add_argument_group('Kernel options')
    g = net.add_mutually_exclusive_group()
    g.add_argument('--net-version', choices=sorted(net_versions),
                   default='nothing',
                   help="How to represent the values before putting them in "
                        "the kernel. Options defined in this file; "
                        "default '%(default)s'.")
    g.add_argument('--net-file',
                   help="A Python file containing a net_custom function "
                        "that does the representation; see existing options "
                        "for examples. (Same API: net_custom(in_p, in_q) "
                        "needs to return net_p, net_q, reg_term.)")
    g = net.add_mutually_exclusive_group(required=True)
    g.add_argument('--max-ratio', '-r',
                   dest='criterion', action='store_const', const='ratio',
                   help="Maximize the t-statistic estimator.")
    g.add_argument('--max-mmd', '-m',
                   dest='criterion', action='store_const', const='mmd',
                   help="Maximize the MMD estimator.")
    g.add_argument('--max-hotelling',
                   dest='criterion', action='store_const', const='hotelling',
                   help="Maximize the Hotelling test statistics; only works "
                        "with a linear kernel.")

    g = net.add_mutually_exclusive_group()
    g.add_argument('--rbf-kernel',
                   default=False, action='store_false', dest='linear_kernel',
                   help="Use an RBF kernel; true by default.")
    g.add_argument('--linear-kernel', default=False, action='store_true')

    g = net.add_mutually_exclusive_group()
    g.add_argument('--biased-est', default=True, action='store_true',
                   help="Use the biased quadratic MMD estimator.")
    g.add_argument('--unbiased-est', dest='biased_est', action='store_false',
                   help="Use the unbiased quadratic MMD estimator.")
    g.add_argument('--streaming-est', default=False, action='store_true',
                   help="Use the streaming estimator for the MMD; faster "
                        "but much less powerful.")

    net.add_argument('--hotelling-reg', type=float, default=0,
                     help="Regularization for the inverse in the Hotelling "
                          "criterion; default %(default)s.")

    g = net.add_mutually_exclusive_group()
    g.add_argument('--opt-sigma', default=False, action='store_true',
                   help="Optimize the bandwidth of an RBF kernel; "
                        "default don't.")
    g.add_argument('--no-opt-sigma', dest='opt_sigma', action='store_false')

    g = net.add_mutually_exclusive_group()
    def context_eval(s):
        return eval(s)
    g.add_argument('--sigma', default='1', type=context_eval,
                   help="The initial bandwidth. Evaluated as Python, so you "
                        "could do e.g. --sigma 'np.random.lognormal()'.")
    g.add_argument('--init-sigma-median', action='store_true', default=False,
                   help="Initialize the bandwidth as the median of pairwise "
                        "distances between representations of the training "
                        "data.")

    g = net.add_mutually_exclusive_group()
    g.add_argument('--opt-log', default=True, action='store_true',
                   help="Optimize the log of the criterion; true by default.")
    g.add_argument('--no-opt-log', dest='opt_log', action='store_false')

    opt = parser.add_argument_group('Optimization')
    opt.add_argument('--num-epochs', type=int, default=10000)
    opt.add_argument('--batchsize', type=int, default=200)
    opt.add_argument('--val-batchsize', type=int, default=1000)
    opt.add_argument('--opt-strat', default='adam')
    opt.add_argument('--learning-rate', type=float, default=.01)
    opt.add_argument('--opt-args', type=ast.literal_eval, default={})

    data = parser.add_argument_group('Data')
    generate.add_problem_args(data)
    data.add_argument('--n-train', type=int, default=500)
    data.add_argument('--n-test', type=int, default=500)

    test = parser.add_argument_group('Testing')
    test.add_argument('--null-samples', type=int, default=1000)

    parser.add_argument('--seed', type=int, default=np.random.randint(2**31))
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--log-params', default=False, action='store_true',
                   help="Log the network parameters at every iteration. Only "
                        "do this if you really want it; parameters are always "
                        "saved at the end.")
    g.add_argument('--no-log-params', dest='log_params', action='store_false')
    parser.add_argument('outfile',
                        help="Where to store the npz file of results.")

    args = parser.parse_args()
    if args.linear_kernel and (args.opt_sigma or args.sigma != 1):
        parser.error("Linear kernel and sigma are incompatible")
    if (not args.linear_kernel) and args.criterion == 'hotelling':
        parser.error("Hotelling criterion only available for linear kernel")

    n_train = args.n_train
    n_test = args.n_test
    np.random.seed(args.seed)
    X, Y = generate.generate_data(args, n_train + n_test, dtype=floatX)
    is_train = np.zeros(n_train + n_test, dtype=bool)
    is_train[np.random.choice(n_train + n_test, n_train, replace=False)] = True
    X_train = X[is_train]
    Y_train = Y[is_train]
    X_test = X[~is_train]
    Y_test = Y[~is_train]

    if args.net_file:
        # This should be a python file with a net_custom function taking in
        # in_p, in_q and return net_p, net_q, reg_term.
        with open(args.net_file) as f:
            code = f.read()
        register_custom_net(code)
        args.net_version = 'custom'
        args.net_code = code

    params, param_names, get_rep, value_log, sigma = train(
        X_train, Y_train, X_test, Y_test,
        criterion=args.criterion,
        biased=args.biased_est,
        hotelling_reg=args.hotelling_reg,
        streaming_est=args.streaming_est,
        linear_kernel=args.linear_kernel,
        opt_log=args.opt_log,
        init_log_sigma=np.log(args.sigma),
        init_sigma_median=args.init_sigma_median,
        opt_sigma=args.opt_sigma,
        net_version=args.net_version,
        num_epochs=args.num_epochs,
        batchsize=args.batchsize,
        val_batchsize=args.val_batchsize,
        opt_strat=args.opt_strat,
        log_params=args.log_params,
        **args.opt_args)

    print("Testing...", end='')
    sys.stdout.flush()
    try:
        p_val, stat, null_samps = eval_rep(
            get_rep, X_test, Y_test,
            linear_kernel=args.linear_kernel, sigma=sigma,
            hotelling=args.criterion == 'hotelling',
            null_samples=args.null_samples)
        print("p-value: {}".format(p_val))
    except ImportError as e:
        print()
        print("Couldn't import shogun:\n{}".format(e), file=sys.stderr)
        p_val, stat, null_samps = None, None, None

    to_save = dict(
       null_samps=null_samps,
       test_stat=stat,
       sigma=sigma,
       p_val=p_val,
       params=params,
       param_names=param_names,
       X_train=X_train, X_test=X_test,
       Y_train=Y_train, Y_test=Y_test,
       value_log=value_log,
       args=args)

    try:
        dirname = os.path.dirname(args.outfile)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)
        np.savez(args.outfile, **to_save)
    except Exception as e:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            name = tmp.name
        msg = "Couldn't save to {}:\n{}\nSaving to {} instead"
        print(msg.format(args.outfile, e, name), file=sys.stderr)
        np.savez(name, **to_save)


if __name__ == '__main__':
    main()
