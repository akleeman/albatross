import os
import re
import sys
from scipy import optimize
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')


"""
To generate benchmark plots you can run,

    for f in 10 7 5 3 2 1; do ./examples/temperature_example -input ../examples/temperature_example/gsod.csv -predict ../examples/temperature_example/prediction_locations.csv -output ./full_predictions.csv -thin $f; done > benchmark.out
    
    python produce_benchmark_plots.py benchmark.out
"""

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    return p

def is_first_line(line):
    return 'Reading' in line

def get_number_of_obs(line):
    assert line.startswith("Using")
    return int(line.strip().split(' ')[1])

def get_fit_time(line):
    assert line.startswith("Fit")
    return float(line.strip().rsplit(' ', 1)[1])

def get_serialize_time(line):
    assert line.startswith("Serializing")
    return float(line.strip().rsplit(' ', 1)[1])

def get_query_time(line):
    assert line.startswith("Each query")
    return float(line.strip().rsplit(' ', 1)[1])

def iter_by_size(lines):
    iterlines = iter(lines)
    while True:
        iterlines.next()
        n = get_number_of_obs(iterlines.next())
        iterlines.next()
        iterlines.next()
        fit_time = get_fit_time(iterlines.next())
        iterlines.next()
        serialize_time = get_serialize_time(iterlines.next())
        print iterlines.next()
        query_time = get_query_time(iterlines.next())

        serialize_size = os.path.getsize('../../build/serialized_%d.model.gz' % n)

        yield {'n': n, 'Fit Time(s)': fit_time,
               'Serialize Time(s)': serialize_time,
               'Query Time(s)': query_time,
               'Serialize Size(MB)': serialize_size / 1.e6}

def power_func(xs, a, b, c):
    return a + b * np.power(xs, c)

def get_fit(x, y):

    def error(params):
        a, b, c = params
        return np.sqrt(np.square(np.log(power_func(x, a, b, c)) - np.log(y)).mean())

    opt = optimize.fmin(error, [0., 1., 1.])

    def best_fit(xs):
        return power_func(xs, *opt)

    return opt, best_fit

if __name__ == "__main__":

    p = create_parser()
    args = p.parse_args()

    with open(args.input, 'r') as f:
        lines = f.readlines()
        
    df = pd.DataFrame(list(iter_by_size(lines)))
    df.set_index('n', inplace=True)
    df.sort_index(inplace=True)

    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    axes = axes.reshape(-1)
    plt.rc('text', usetex=True)

    for v, ax in zip(df.keys(), axes):
        params, fit = get_fit(df.index.values, df[v].values)
        ax.loglog(df.index.values, df[v].values, 'k.')   
        xs = np.logspace(2.5, 5, 101)
        ys = fit(xs)
        ax.loglog(xs, ys)
        a, b = re.split('e-[0]*', '%.1e' % params[1])
        latex_text = "$%se^{-%s} n^{%3.1f}$" % (a, b, params[2])
        ax.text(0.7, 0.2, latex_text,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=16,
                transform=ax.transAxes)
        ax.set_title("%s versus N observations" % v)
    fig.tight_layout()
    plt.show()

    import ipdb; ipdb.set_trace()
    print "done"
