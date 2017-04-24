import os
import sys

import numpy as np
from sklearn.utils import check_random_state


################################################################################
### Simple toy problems

def sample_SG(n, dim, rs=None):
    rs = check_random_state(rs)
    mu = np.zeros(dim)
    sigma = np.eye(dim)
    X = rs.multivariate_normal(mu, sigma, size=n)
    Y = rs.multivariate_normal(mu, sigma, size=n)
    return X, Y


def sample_GMD(n, dim, rs=None):
    rs = check_random_state(rs)
    mu = np.zeros(dim)
    sigma = np.eye(dim)
    X = rs.multivariate_normal(mu, sigma, size=n)
    mu[0] += 1
    Y = rs.multivariate_normal(mu, sigma, size=n)
    return X, Y


def sample_GVD(n, dim, rs=None):
    rs = check_random_state(rs)
    mu = np.zeros(dim)
    sigma = np.eye(dim)
    X = rs.multivariate_normal(mu, sigma, size=n)
    sigma[0, 0] = 2
    Y = rs.multivariate_normal(mu, sigma, size=n)
    return X, Y


def sample_blobs(n, ratio, rows=5, cols=5, sep=10, rs=None):
    rs = check_random_state(rs)
    # ratio is eigenvalue ratio
    correlation = (ratio - 1) / (ratio + 1)

    # generate within-blob variation
    mu = np.zeros(2)
    sigma = np.eye(2)
    X = rs.multivariate_normal(mu, sigma, size=n)

    corr_sigma = np.array([[1, correlation], [correlation, 1]])
    Y = rs.multivariate_normal(mu, corr_sigma, size=n)

    # assign to blobs
    X[:, 0] += rs.randint(rows, size=n) * sep
    X[:, 1] += rs.randint(cols, size=n) * sep
    Y[:, 0] += rs.randint(rows, size=n) * sep
    Y[:, 1] += rs.randint(cols, size=n) * sep

    return X, Y


################################################################################
### Sample images from GANs

def _load_mnist(dset='t10k'):
    # Basically taken from Lasagne/examples/mnist.py
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source="http://yann.lecun.com/exdb/mnist/"):
        print("Downloading {}".format(filename))
        urlretrieve(source + filename, filename)

    import gzip
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(255)

    return load_mnist_images(dset + '-images-idx3-ubyte.gz')


def _sample_trained_minibatch_gan(params_file, n, batch_size, rs):
    import lasagne
    from lasagne.init import Normal
    import lasagne.layers as ll
    import theano as th
    from theano.sandbox.rng_mrg import MRG_RandomStreams
    import theano.tensor as T

    import nn

    theano_rng = MRG_RandomStreams(rs.randint(2 ** 15))
    lasagne.random.set_rng(np.random.RandomState(rs.randint(2 ** 15)))

    noise_dim = (batch_size, 100)
    noise = theano_rng.uniform(size=noise_dim)
    ls = [ll.InputLayer(shape=noise_dim, input_var=noise)]
    ls.append(nn.batch_norm(
        ll.DenseLayer(ls[-1], num_units=4*4*512, W=Normal(0.05),
                      nonlinearity=nn.relu),
        g=None))
    ls.append(ll.ReshapeLayer(ls[-1], (batch_size,512,4,4)))
    ls.append(nn.batch_norm(
        nn.Deconv2DLayer(ls[-1], (batch_size,256,8,8), (5,5), W=Normal(0.05),
                         nonlinearity=nn.relu),
        g=None)) # 4 -> 8
    ls.append(nn.batch_norm(
        nn.Deconv2DLayer(ls[-1], (batch_size,128,16,16), (5,5), W=Normal(0.05),
                         nonlinearity=nn.relu),
        g=None)) # 8 -> 16
    ls.append(nn.weight_norm(
        nn.Deconv2DLayer(ls[-1], (batch_size,3,32,32), (5,5), W=Normal(0.05),
                         nonlinearity=T.tanh),
        train_g=True, init_stdv=0.1)) # 16 -> 32
    gen_dat = ll.get_output(ls[-1])

    with np.load(params_file) as d:
        params = [d['arr_{}'.format(i)] for i in range(9)]
    ll.set_all_param_values(ls[-1], params, trainable=True)

    sample_batch = th.function(inputs=[], outputs=gen_dat)
    samps = []
    while len(samps) < n:
        samps.extend(sample_batch())
    samps = np.array(samps[:n])
    return samps


def sample_mnist_minibatch_gan(
        n, params_file, batch_size=100, rs=None, mnist_images=None,
        discretize=None, bw=False, grayscale=True, clip=True, scaled=False,
        trim_edges=False):
    rs = check_random_state(rs)

    Y = _sample_trained_minibatch_gan(params_file, n, min(n, batch_size), rs)

    if mnist_images is None:
        mnist_images = _load_mnist()

    X = mnist_images[rs.choice(mnist_images.shape[0], n, replace=False), :]

    # X is shape (n, 1, 28, 28); Y is (n, 3, 32, 32)
    # Process them to a common format:

    # GAN images are color, MNIST are grayscale
    if grayscale or bw:
        # 0.2125 R + 0.7154 G + 0.0721 B, per skimage.color.rgb2gray
        Y = np.einsum('nchw,c->nhw', Y, [0.2125, 0.7154, 0.0721])
        X = X[:, 0, :, :]
    else:
        X = np.tile(X, (1, 3, 1, 1))

    # GAN images are 32x32, MNIST are 28x28
    if trim_edges:
        Y = Y[..., 2:-2, 2:-2]
    else:
        t = X
        X = np.zeros(tuple(32 if s == 28 else s for s in t.shape), t.dtype)
        X[..., 2:-2, 2:-2] = t

    # GAN images have range [-1, 1]; MNIST has [0, 1]
    if scaled:
        Y += 1
        Y /= 2
    elif clip:
        np.clip(Y, 0, 1, out=Y)

    # flatten
    X = X.reshape(n, -1)
    Y = Y.reshape(n, -1)

    # pixel-level differences make the problem too easy; maybe discretize
    if bw:
        X = X.round()
        Y = Y.round()
        if not scaled and not clip:
            np.clip(Y, 0, 1, out=Y)
    elif discretize:
        bins = np.linspace(0, 1 + np.spacing(1), num=discretize + 1)
        midpoints = (bins[:-1] + bins[1:]) / 2.

        Y = midpoints[np.digitize(Y, bins) - 1]
        X = midpoints[np.digitize(X, bins) - 1]

    return X, Y


################################################################################
### Helpers to use with argparse

def add_problem_args(group):
    g = group.add_mutually_exclusive_group(required=True)
    g.add_argument('--sg', '--same-gaussian', type=int, metavar='DIM')
    g.add_argument('--gmd', '--gaussian-mean-difference',
                   type=int, metavar='DIM')
    g.add_argument('--gvd', '--gaussian-var-difference',
                   type=int, metavar='DIM')
    g.add_argument('--blobs', type=float, metavar='EIG_RATIO')
    g.add_argument('--mnist-minibatch-gan', metavar='PARAMS_FILE')
    g.add_argument('--mnist-traintest', action='store_true')

    g = group.add_mutually_exclusive_group()
    g.add_argument('--grayscale', action='store_true', default=False,
                   help="For GAN outputs: make images grayscale.")
    g.add_argument('--no-grayscale', action='store_false', dest='grayscale')

    g = group.add_mutually_exclusive_group()
    g.add_argument('--bw', action='store_true', default=False,
                   help="For GAN outputs: make images black+white (implies "
                        "--grayscale).")
    group.add_argument('--discretize', type=int, metavar='N_BINS',
                       help="For GAN outputs: discretize possible outputs "
                            "into N_BINS bins. Note that "
                            "`--grayscale --discretize 2` makes the outputs "
                            "[.25, .75], where `--bw` makes them [0, 1].")

    g = group.add_mutually_exclusive_group()
    g.add_argument('--trim-edges', action='store_true', default=False,
                   help="For MNIST GANs: trim the outer border of samples.")
    g.add_argument('--no-trim-edges', action='store_false', dest='trim_edges')

    g = group.add_mutually_exclusive_group()
    g.add_argument('--clip', action='store_true', default=True,
                   help="For GAN outputs: clip pixel values to [0, 1]. "
                        "On by default.")
    g.add_argument('--no-clip', action='store_false', dest='clip',
                   help="For GAN outputs: leave pixels as they are, possibly "
                        "in [-1, 1].")
    g.add_argument('--scaled', action='store_true', default=False,
                   help="For GAN outputs: scale pixel values to [0, 1].")


def generate_data(args, n, dtype=None, rs=None):
    if args.sg is not None:
        X, Y = sample_SG(n, args.sg, rs=rs)
    elif args.gmd is not None:
        X, Y = sample_GMD(n, args.gmd, rs=rs)
    elif args.gvd is not None:
        X, Y = sample_GVD(n, args.gvd, rs=rs)
    elif args.blobs is not None:
        X, Y = sample_blobs(n, args.blobs, rs=rs)
    elif args.mnist_minibatch_gan is not None:
        X, Y = sample_mnist_minibatch_gan(
            n, args.mnist2_gan, rs=rs, grayscale=args.grayscale, bw=args.bw,
            trim_edges=args.trim_edges, clip=args.clip, scaled=args.scaled,
            discretize=args.discretize)
    elif args.mnist_traintest:
        rs = check_random_state(rs)
        # MNIST loads as n x 1 x 28 x 28; want n x 784
        X = _load_mnist('t10k').reshape(-1, 784)
        X = X[rs.choice(X.shape[0], n, replace=False), :]
        Y = _load_mnist('train').reshape(-1, 784)
        Y = Y[rs.choice(Y.shape[0], n, replace=False), :]
    else:
        raise ValueError("No dataset passed")

    if dtype is not None:
        X = X.astype(dtype)
        Y = Y.astype(dtype)
    return X, Y

