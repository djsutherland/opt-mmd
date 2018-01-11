from __future__ import division, print_function
from collections import OrderedDict
from itertools import chain
from functools import partial
import multiprocessing as mp
import os
import sys

try:
    import modshogun as sg
except ImportError:  # new versions just call it shogun
    import shogun as sg

import numpy as np
import pandas as pd
import progressbar as pb
from six.moves import xrange
from scipy.stats.mstats import mquantiles

import generate


if 'OMP_NUM_THREADS' in os.environ:
    num_threads = int(os.environ['OMP_NUM_THREADS'])
else:
    num_threads = mp.cpu_count()
sg.get_global_parallel().set_num_threads(num_threads)


def get_estimates(gen, sigmas=None, n_reps=100, n_null_samps=1000,
                  cache_size=64, rep_states=False, name=None,
                  save_samps=False, thresh_levels=(.2, .1, .05, .01)):
    if sigmas is None:
        sigmas = np.logspace(-1.7, 1.7, num=30)
    sigmas = np.asarray(sigmas)

    mmd = sg.QuadraticTimeMMD()
    mmd.set_num_null_samples(n_null_samps)
    mmd_mk = mmd.multikernel()
    for s in sigmas:
        mmd_mk.add_kernel(sg.GaussianKernel(cache_size, 2 * s**2))

    info = OrderedDict()
    for k in 'sigma rep mmd_est var_est p'.split():
        info[k] = []
    thresh_names = []
    for l in thresh_levels:
        s = 'thresh_{}'.format(l)
        thresh_names.append(s)
        info[s] = []
    if save_samps:
        info['samps'] = []

    thresh_prob = 1 - np.asarray(thresh_levels)

    bar = pb.ProgressBar()
    if name is not None:
        bar.start()
        bar.widgets.insert(0, '{} '.format(name))
    for rep in bar(xrange(n_reps)):
        if rep_states:
            rep = np.random.randint(0, 2**32)
            X, Y = gen(rs=rep)
        else:
            X, Y = gen()
        n = X.shape[0]
        assert Y.shape[0] == n
        mmd.set_p(sg.RealFeatures(X.T))
        mmd.set_q(sg.RealFeatures(Y.T))

        info['sigma'].extend(sigmas)
        info['rep'].extend([rep] * len(sigmas))

        stat = mmd_mk.compute_statistic()
        info['mmd_est'].extend(stat / (n / 2))

        samps = mmd_mk.sample_null()
        info['p'].extend(np.mean(samps >= stat, axis=0))
        if save_samps:
            info['samps'].extend(samps.T)

        info['var_est'].extend(mmd_mk.compute_variance_h1())

        threshes = np.asarray(mquantiles(samps, prob=thresh_prob, axis=0))
        for s, t in zip(thresh_names, threshes):
            info[s].extend(t)

    info = pd.DataFrame(info)
    info.set_index(['sigma', 'rep'], inplace=True)
    return info


def try_int(x):
    try:
        i = int(x)
    except (TypeError, ValueError):
        return x
    else:
        return i if i == x else x


def main():
    import argparse
    parser = argparse.ArgumentParser(description='''
        Evaluate different kernel selection methods for picking the bandwidth
        of an RBF kernel.

        Makes an hdf5 file containing results in OUT_FILE; see fixed_eval.py
        to make a jupyter notebook evaluating the results.
        '''.strip())

    data = parser.add_argument_group('Data')
    generate.add_problem_args(data)
    data.add_argument('-n', type=int, default=500)

    parser.add_argument('--n-reps', type=int, default=100)
    parser.add_argument('--n-null-samps', type=int, default=1000)

    parser.add_argument('--sigma-base', type=float, default=1)
    parser.add_argument('--sigma-max-mult', type=float, default=10**1.7)
    parser.add_argument('--n-sigmas', type=int, default=30)

    io = parser.add_argument_group('I/O')
    io.add_argument('out_file', nargs='?',
                    help="Defaults to res_fixed/PROBLEM/(d|rat)ARG/nN.h5.")
    io.add_argument('out_key', nargs='?', default='df')

    g = io.add_mutually_exclusive_group()
    g.add_argument('--save-samps', action='store_true', default=False)
    g.add_argument('--no-save-samps', action='store_false', dest='save_samps')

    io.add_argument('--levels', type=float, nargs='*',
                        default=[.2, .1, .05, .01])

    args = parser.parse_args()
    if args.out_file is None:
        name = next(chain(
            ('{}/d{}/n{}'.format(base, try_int(d), args.n) for base, d
              in [('sg', args.sg), ('gmd', args.gmd), ('gvd', args.gvd)]
              if d is not None),
            ('{}/rat{}/n{}'.format(base, try_int(r), args.n) for base, r
             in [('blobs', args.blobs)]
             if r is not None),
        ))
        args.out_file = os.path.join(
            os.path.dirname(__file__), 'res_fixed/{}.h5'.format(name))
    else:
        name = os.path.relpath(args.out_file)
        if args.out_key != 'df':
            name += ':{}'.format(args.out_key)

    d = os.path.dirname(args.out_file)
    if d and not os.path.isdir(d):
        os.makedirs(d)

    v = np.log10(args.sigma_max_mult)
    sigmas = np.logspace(-v, v, num=args.n_sigmas) * args.sigma_base

    gen = partial(generate.generate_data, args, args.n, dtype=np.float64)
    info = get_estimates(
        gen, n_reps=args.n_reps, n_null_samps=args.n_null_samps,
        rep_states=True, name=name, sigmas=sigmas,
        thresh_levels=args.levels, save_samps=args.save_samps)

    if args.save_samps:
        # for serialization, need to convert samps to a string :|
        info['samps'] = info.samps.map(lambda x: x.tostring())

    try:
        info.to_hdf(args.out_file, args.out_key,
                    format='table', complevel=9, complib='blosc', append=True)
    except Exception as e:
        if sys.stdin.isatty() and sys.stderr.isatty():
            print(e, file=sys.stderr)
            import IPython; IPython.embed()
        else:
            raise


if __name__ == '__main__':
    main()
