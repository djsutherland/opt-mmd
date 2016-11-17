from __future__ import division, print_function
import os
import re

import nbformat as nbf
from nbconvert.nbconvertapp import NbConvertApp


################################################################################
### Making report notebooks

def make_notebook(out_name, fname, key, n, level=.1, kernel_name=None):
    v = nbf.v4
    nb = v.new_notebook()
    if kernel_name is not None:
        from jupyter_client.kernelspec import KernelSpecManager
        d = KernelSpecManager().get_kernel_spec(kernel_name).to_dict()
        nb['metadata']['kernelspec'] = {
            'name': kernel_name,
            'display_name': d['display_name'],
        }
        nb['metadata']['language_info'] = {
            'name': d['language'],
        }

    add_code = lambda s: nb['cells'].append(v.new_code_cell(s.strip()))
    add_markdown = lambda s: nb['cells'].append(v.new_markdown_cell(s.strip()))

    add_code(r'''
from __future__ import division, print_function
from operator import itemgetter
import os

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats.mstats import mquantiles
from scipy import stats
''')

    add_code(r'''
probs = (.025, .16, .5, .84, .975)
def plot_band(group, probs=probs, ax=None, **kwargs):
    if ax is None:
        from matplotlib import pyplot as ax
    x = np.asarray([k for k, v in group])
    assert np.all(np.diff(x) > 0)
    bits = group.apply(lambda y: stats.mstats.mquantiles(y, prob=probs))
    get = lambda i: bits.apply(itemgetter(i))
    ax.fill_between(x, get(0), get(-1), alpha=.2, **kwargs)
    ax.fill_between(x, get(1), get(-2), alpha=.2, **kwargs)
    ax.plot(x, get(2))
''')

    add_code(r'''
basedir = {!r}
fname = {!r}
key = {!r}
n = {!r}
level = {!r}
'''.format(os.path.abspath(os.path.dirname(__file__)), fname, key, n, level))

    add_markdown('## Load data')
    add_code(r'''
# load data
d = pd.read_hdf(os.path.join(basedir, fname), key)
if 'samps' in d:
    d['samps'] = d.samps.map(lambda x: np.fromstring(x, dtype=np.float32))
''')

    add_code(r'''
# basic info
sigmas = d.index.get_level_values('sigma').unique()
assert np.all(np.diff(sigmas) > 0)

n_reps = len(d.index.get_level_values('rep').unique())
''')

    add_markdown("Estimate the power for each bandwidth by the number of "
                 "successes across repetitions:")
    add_code(r'''
n_succ = d.groupby(level='sigma').p.agg(lambda x: (x <= level).sum())
power = n_succ / n_reps
''')

    add_code(r'''
m = 'jeffrey'
lo1, hi1 = sm.stats.proportion_confint(n_succ, n_reps, alpha=.32, method=m)
lo2, hi2 = sm.stats.proportion_confint(n_succ, n_reps, alpha=.05, method=m)

plt.xscale('log')
plt.fill_between(sigmas, lo2, hi2, alpha=.2)
plt.fill_between(sigmas, lo1, hi1, alpha=.2)
plt.plot(sigmas, power)
plt.ylim(0, 1)
''')

    add_code(r'''
def add_power(ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    kwargs.setdefault('color', 'k')
    kwargs.setdefault('ls', '--')
    kwargs.setdefault('alpha', .5)

    xlim = ax.get_xlim()
    ax2 = ax.twinx()
    ax2.plot(sigmas, power, **kwargs)
    ax2.grid(False)
    ax2.set_yticks([], [])
    ax2.set_ylim(0, 1)
    ax2.set_xlim(*xlim)
''')

    add_markdown("## Computing criteria")
    add_code(r'''
d['ratio'] = d.mmd_est / np.sqrt(d.var_est)
if 'thresh_{}'.format(level) in d:
    d['thresh'] = d['thresh_{}'.format(level)]
elif 'samps' in d:
    d['thresh'] = d.samps.map(lambda s: mquantiles(s, prob=1 - level)[0])
else:
    raise ValueError("Sorry, can't do level {}. Could do any of {}".format(
        level, ', '.join(str(float(k[len('thresh_'):]))
                         for k in d if k.startswith('thresh_'))))
d['criterion'] = -(d.thresh / (n * np.sqrt(d.var_est)) - d.ratio)
''')

    add_code(r'''
plt.xscale('log')
plot_band(d.mmd_est.groupby(level=0))
plt.title(r'MMD$^2$ estimates')
add_power()
''')

#     add_code(r'''
# plt.xscale('log')
# plot_band(d.var_est.apply(np.sqrt).groupby(level=0))
# plt.title(r'$\sqrt{\text{Var}}$ estimates')
# add_power()
# ''')

    add_code(r'''
plt.xscale('log')
plot_band(d.ratio.groupby(level=0))
plt.title(r'$\mathrm{MMD}^2 / \sqrt{\mathrm{Var}}$ estimates')
add_power()
''')

    add_code(r'''
plt.xscale('log')
plot_band(d.criterion.groupby(level=0))
plt.title(r'$-(\frac{c_\alpha}{n \sqrt{\mathrm{Var}}} - '
          r'\frac{\mathrm{MMD}^2}{\sqrt{\mathrm{Var}}})$ estimates')
add_power()
''')

    add_markdown(r"## Maximizing estimates")
    add_code(r'''
max_sigmas = d[['mmd_est', 'ratio', 'criterion']].groupby(level='rep').agg(lambda x: x.idxmax()[0])
max_powers = max_sigmas.apply(lambda x: power.loc[x].values)
''')

    add_markdown(r"Selected bandwidths by each method:")
    add_code(r'''
for i, (k, color) in enumerate([('mmd_est', 'b'), ('ratio', 'r'), ('criterion', 'g')], 1):
    ax = plt.subplot(3, 1, i)

    for s in sigmas:
        ax.axvline(s, color='w')

    vc = max_sigmas[k].value_counts().loc[sigmas].fillna(0)
    for s, c in vc.iteritems():
        c = int(c)
        ax.plot([s] * c, np.arange(c),
                 marker='o', ls='', color=color)

    ax.set_xscale('log')
    ax.grid(False)
    ax.set_yticks([], [])
    ax.set_xlim(sigmas[0] / 1.1, sigmas[-1] * 1.1)
    ax.set_ylim(-1, vc.max())
    ax.set_title(k)
    add_power(ax)
plt.tight_layout()
#**chosen_sigmas_stacked** <- keep this tag for get_plots
''')

    add_code(r'''
plt.xscale('log')
xs = np.linspace(np.log10(sigmas[0]) - .1, np.log10(sigmas[-1]) + .1, 50)
for k in max_sigmas.columns:
    kde = stats.gaussian_kde(max_sigmas[k].apply(np.log10))
    plt.plot(10 ** xs, kde(xs), label=k)
plt.legend()

add_power()
#**chosen_sigmas_kde** <- keep this tag for get_plots
''')

    add_markdown(r"Powers of the bandwidths selected by each method:")
    add_code(r'''
a = np.array(max_powers).T
a += np.random.uniform(-1e-6, 1e-6, size=a.shape)  # jitter so KDE still works
sm.graphics.beanplot(a, labels=max_powers.columns, jitter=True)

mx = power.max()
plt.axhline(mx, color='k', alpha=.5)
plt.ylim(0, min(1, mx + .05, mx * 1.1))
#**power_beanplots** <- keep this tag for get_plots
''')

    with open(out_name, 'w') as f:
        nbf.write(nb, f)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='''
        Make (and run) a jupyter notebook visualizing the results of an output
        file from fixed_run.py.'''.strip())
    parser.add_argument('fname')
    parser.add_argument('--key', default='df')
    parser.add_argument('out_name', nargs='?')
    parser.add_argument('-n', type=int)
    parser.add_argument('--level', type=float, default=.1)
    parser.add_argument('--kernel-name')
    parser.add_argument('--force', '-f', action='store_true')
    args = parser.parse_args()

    if args.out_name is None:
        assert args.fname.endswith('.h5')
        base = args.fname[:-3]
        args.out_name = base + '.ipynb'
    else:
        d = os.path.dirname(args.out_name)
        if d and not os.path.isdir(d):
            os.makedirs(d)

    if not args.force and os.path.exists(args.out_name):
        parser.exit("Output {} already exists; use --force to overwrite."
                    .format(args.out_name))
    del args.force

    if args.n is None:
        match = re.search('/n(\d+)\.h5$', args.fname)
        if match:
            args.n = int(match.group(1))
        else:
            parser.error('-n must be specified unless filename has it')

    make_notebook(**vars(args))
    nbc = NbConvertApp()
    nbc.initialize([
        '--to=notebook', '--execute',
        '--output', os.path.basename(args.out_name),
        args.out_name])
    nbc.convert_notebooks()


if __name__ == '__main__':
    main()
