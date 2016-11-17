from __future__ import division, print_function
import sys

import numpy as np
import lasagne
import theano
import theano.tensor as T

import learn_kernel
import generate

try:
    import progressbar as pb
    have_pb = True
except ImportError:
    have_pb = False


def load_network(results_file, n_test=None):
    with np.load(results_file) as d:
        args = d['args'][()]
        params = d['params'] if 'params' in d else d['net_params']
        sigma_val = d['sigma'][()]

    def gen_data(n=None, dtype=learn_kernel.floatX):
        return generate.generate_data(args, n or args.n_test, dtype=dtype)

    if getattr(args, 'net_code', False):
        learn_kernel.register_custom_net(args.net_code)

    # make the representation network; don't bother calling make_network since
    # it does a bunch of other things
    dim = gen_data()[0].shape[1]  # could be smarter about this
    input_p = T.matrix('input_p')
    input_q = T.matrix('input_q')
    in_p = lasagne.layers.InputLayer(shape=(None, dim), input_var=input_p)
    in_q = lasagne.layers.InputLayer(shape=(None, dim), input_var=input_q)
    net_p, net_q, reg = learn_kernel.net_versions[args.net_version](in_p, in_q)
    rep_p = lasagne.layers.get_output(net_p)

    print("Compiling...", file=sys.stderr, end='')
    get_rep = theano.function([input_p], rep_p)
    print("done.", file=sys.stderr)

    if getattr(args, 'opt_sigma', False):
        params = params[:-1]
    lasagne.layers.set_all_param_values(net_p, params)

    return get_rep, gen_data, sigma_val, args.linear_kernel


def eval_network(get_rep, gen_data, n_reps=10, null_samples=1000, n_test=None,
                 sigma=1, linear_kernel=False, hotelling=True):
    t = range(n_reps)
    if have_pb:
        t = pb.ProgressBar()(t)
    for i in t:
        X, Y = gen_data(n=n_test, dtype=np.float32)
        p_val, _, _ = learn_kernel.eval_rep(
            get_rep, X, Y, linear_kernel=linear_kernel, sigma=sigma,
            hotelling=hotelling, null_samples=null_samples)
        yield p_val


def main():
    import argparse
    parser = argparse.ArgumentParser(description='''
        Evaluate a learned representation (from learn_kernel.py) on
        repeatedly-generated test data; print out the p-value of the test on 
        each set we try.'''.strip())
    parser.add_argument('filename')
    parser.add_argument('--n-reps', type=int, default=10,
                        help="How many test sets to evaluate on (default "
                             "%(default)s).")
    parser.add_argument('--null-samples', type=int, default=1000,
                        help="How many null samples to take (default "
                             "%(default)s).")
    parser.add_argument('--n-test', type=int, default=None,
                        help="How big of a test set to use.")

    g = parser.add_mutually_exclusive_group()
    g.add_argument('--hotelling', action='store_true', default=True,
                   help="If it's a linear kernel, use the Hotelling test "
                        "(default behavior).")
    g.add_argument('--permutation', action='store_false', dest='hotelling',
                   help="Do the permutation test even if it's a linear kernel.")
    args = parser.parse_args()

    get_rep, gen_data, sigma, linear_kernel = load_network(args.filename)
    p_vals = eval_network(
            get_rep, gen_data, n_reps=args.n_reps,
            n_test=args.n_test, null_samples=args.null_samples,
            sigma=sigma, linear_kernel=linear_kernel, hotelling=args.hotelling)
    for p in p_vals:
        if 0 <= p < 1:
            print('{:.3f}'.format(p)[1:])
        else:
            print('{:.2f}'.format(p))
    print()

if __name__ == '__main__':
    main()
