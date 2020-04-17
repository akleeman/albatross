import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy import stats

sns.set_style('darkgrid')


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--thin", type=int, default=10)
    p.add_argument("--burn-in", type=int, default=100)
    p.add_argument("--variables", type=str, default=None, nargs="*")
    p.add_argument("--mode", choices=["corner", "plume", "list", "progress", "one_vs_all"], default="corner")
    return p


def one_vs_all_plot(args):
    df = pd.read_csv(args.input);
    for k in df.columns:
        df[k] = df[k].astype('f')
    df = df.iloc[np.all(np.isfinite(df.values), axis=1)]

    samples = [v for k, v in df.groupby('iteration')]
    
    if args.burn_in > len(samples):
        raise ValueError("burn in exceeds the number of samples")

    samples = samples[args.burn_in:]
    samples = samples[::args.thin]

    columns = np.setdiff1d(samples[0].columns, ['iteration', 'ensemble_index'])

    if args.variables:
        for var_name in args.variables:
            if var_name not in columns:
                print("requested variable %s is not valid" % var_name)
        columns = args.variables

    data = np.vstack([s[columns].values.astype('f') for s in samples])
    log_p = np.concatenate([s["log_probability"].values for s in samples])
    log_p = np.ma.masked_array(log_p, mask=~np.isfinite(log_p))

    iteration = np.concatenate([s["iteration"].values for s in samples])
    good = log_p >= stats.mstats.mquantiles(log_p, [0.1])

    def get_limits(values):
        low, high = stats.mstats.mquantiles(values, [0.05, 0.95])
        d = 0.05 * (high - low)
        if d == 0.:
            d = 1e-4
        low = max(np.min(values), low - d)
        high = min(np.max(values), high + d)
        return (low, high)

    limits = [get_limits(vals) for vals in data[good].T]

    def get_yscale(low, high):
        if abs(high / max(1e-8, low)) > 100:
            return 'log'
        else:
            return 'linear'

    yscales = [get_yscale(low, high) for low, high in limits]

    def get_bins(low, high, yscale):
        if yscale == 'log':
            return np.logspace(np.log10(low), np.log10(high), 30)
        else:
            return np.linspace(low, high, 20)


    bins = [get_bins(lim[0], lim[1], yscale) for lim, yscale in zip(limits, yscales)]

    k = data.shape[1]
    m = int(np.floor(np.sqrt(k)))
    n = int(np.ceil(k / m))

    for i in range(len(columns)):

        fig = plt.figure(figsize=(20, 16))
        grid = plt.GridSpec(m, n)

        for j in range(len(columns)):

            if i == j:
                ax_kwdargs = {}
            else:
                ax_kwdargs = {"xticks": [], "yticks": []}

            row = np.mod(j, m)
            col = int((j - row) / m)
            ax = fig.add_subplot(grid[row, col], **ax_kwdargs)

            if i == j:
                ax.hist(data[:, i], bins=bins[i], color='steelblue')
                ax.set_xscale(yscales[i])
            else:
                ax.hist2d(data[:, i], data[:, j], bins=[bins[i], bins[j]], norm=mcolors.PowerNorm(0.5))
                ax.set_yscale(yscales[j])
                ax.set_xscale(yscales[i])
                ax.set_ylim(limits[j])
                ax.set_xlim(limits[i])
                ax.set_ylabel(columns[j])
        fig.suptitle(columns[i])
        grid.tight_layout(fig)
        os.makedirs("./one_vs_all", exist_ok=True)
        plt.savefig(os.path.join("one_vs_all", columns[i] + ".png"))
        plt.close()
        print(columns[i])


def corner_plot(args):
    df = pd.read_csv(args.input);
    for k in df.columns:
        df[k] = df[k].astype('f')
    df = df.iloc[np.all(np.isfinite(df.values), axis=1)]

    samples = [v for k, v in df.groupby('iteration')]
    
    if args.burn_in > len(samples):
        raise ValueError("burn in exceeds the number of samples")

    samples = samples[args.burn_in:]
    samples = samples[::args.thin]

    columns = np.setdiff1d(samples[0].columns, ['iteration', 'ensemble_index'])

    if args.variables:
        for var_name in args.variables:
            if var_name not in columns:
                print("requested variable %s is not valid" % var_name)
        columns = args.variables

    data = np.vstack([s[columns].values.astype('f') for s in samples])
    log_p = np.concatenate([s["log_probability"].values for s in samples])
    log_p = np.ma.masked_array(log_p, mask=~np.isfinite(log_p))

    iteration = np.concatenate([s["iteration"].values for s in samples])
    good = log_p >= stats.mstats.mquantiles(log_p, [0.1])

    def get_limits(values):
        low, high = stats.mstats.mquantiles(values, [0.05, 0.95])
        d = 0.05 * (high - low)
        if d == 0.:
            d = 1e-4
        low = max(np.min(values), low - d)
        high = min(np.max(values), high + d)
        return (low, high)

    limits = [get_limits(vals) for vals in data[good].T]

    def get_yscale(low, high):
        if abs(high / max(1e-8, low)) > 100:
            return 'log'
        else:
            return 'linear'

    yscales = [get_yscale(low, high) for low, high in limits]

    def get_bins(low, high, yscale):
        if yscale == 'log':
            return np.logspace(np.log10(low), np.log10(high), 30)
        else:
            return np.linspace(low, high, 20)


    bins = [get_bins(lim[0], lim[1], yscale) for lim, yscale in zip(limits, yscales)]

    k = data.shape[1]
    fig = plt.figure(figsize=(16, 16))

    if k > 1:
        grid = plt.GridSpec(k, k, hspace=0.2, wspace=0.2)
    else:
        grid = plt.GridSpec(1, 2, hspace=0.2, wspace=0.2)

    for i in range(len(columns)):
        for j in range(i + 1):
            if i == j or i == len(columns) - 1:
                ax_kwdargs = {}
            else:
                ax_kwdargs = {"xticks": []}

            ax = fig.add_subplot(grid[i, j], **ax_kwdargs)

            if i == j:
                ax.hist(data[:, i], bins=bins[i], color='steelblue')
                ax.set_xscale(yscales[i])
            else:
                ax.hist2d(data[:, j], data[:, i], bins=[bins[j], bins[i]], norm=mcolors.PowerNorm(0.5))
                ax.set_yscale(yscales[i])
                ax.set_xscale(yscales[j])
                ax.set_ylim(limits[i])
                ax.set_xlim(limits[j])
            if i == k - 1:
                ax.set_xlabel(columns[j])
            if j == 0:
                ax.set_ylabel(columns[i])

    if k > 2:
        mid = int(np.ceil(k / 3))
        ax = fig.add_subplot(grid[:mid, (mid + 1):], xticklabels=[])
    else:
        ax = fig.add_subplot(grid[0, 1], xticklabels=[])

    ax.plot(iteration, log_p, '.')
    ylim = stats.mstats.mquantiles(log_p, [0.01, 1.])
    ylim_buffer = 0.05 * (ylim[1] - ylim[0])
    ylim[0] -= ylim_buffer
    ylim[1] += ylim_buffer
    ax.set_ylim(ylim)  
    ax.set_ylabel("log probability")
    ax.set_xlabel("iterations")  
    x_ticks = np.linspace(np.min(iteration), np.max(iteration), 11)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(map(lambda x : str(int(x)), x_ticks))
    plt.show()


def plume_plot(args):
    df = pd.read_csv(args.input);
    for k in df.columns:
        df[k] = df[k].astype('f')
    df = df.iloc[np.all(np.isfinite(df.values), axis=1)]

    samples = [v for k, v in df.groupby('iteration')]
    samples = samples[args.burn_in:]
    samples = samples[::args.thin]

    columns = np.setdiff1d(samples[0].columns, ['iteration', 'log_probability', 'ensemble_index'])

    if args.variables:
        for var_name in args.variables:
            if var_name not in columns:
                print("requested variable %s is not valid" % var_name)
        columns = args.variables

    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 12))
    if len(columns) == 1:
        axes = [axes]

    for ax, var_name in zip(axes, columns):
        data = np.concatenate([s[var_name].values for s in samples])
        log_p = np.concatenate([s["log_probability"].values for s in samples])
        log_p = np.ma.masked_array(log_p, mask=~np.isfinite(log_p))
        iteration = np.concatenate([s["iteration"].values for s in samples])
    
        ax.plot(iteration, np.log10(data), '.')
        ax.set_ylabel(var_name)
        ax.set_xlabel("iterations")  
        x_ticks = np.linspace(np.min(iteration), np.max(iteration), 11)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(map(lambda x : str(int(x)), x_ticks))
    plt.show()


def get_progress_data(args):
    df = pd.read_csv(args.input);
    for k in df.columns:
        df[k] = df[k].astype('f')
    df = df.iloc[np.all(np.isfinite(df.values), axis=1)]

    samples = [v for k, v in df.groupby('iteration')]
    samples = samples[args.burn_in:]
    samples = samples[::args.thin]

    log_p = np.concatenate([s["log_probability"].values for s in samples])
    log_p = np.ma.masked_array(log_p, mask=~np.isfinite(log_p))

    iteration = np.concatenate([s["iteration"].values for s in samples])
    return iteration, log_p

def progress_plot(args):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    points = ax.plot(*get_progress_data(args), '.')[0]

    plt.show(False)
    plt.draw()

    while True:
        iteration, log_p = get_progress_data(args)
        points.set_data(iteration, log_p)

        ylim = stats.mstats.mquantiles(log_p, [0.01, 1.])
        ylim_buffer = 0.05 * (ylim[1] - ylim[0])
        ylim[0] -= ylim_buffer
        ylim[1] += ylim_buffer
        ax.set_xlim([0, np.max(iteration) + 2])
        ax.set_ylim(ylim)  
        ax.set_ylabel("log probability")
        ax.set_xlabel("iterations")  
        x_ticks = np.linspace(np.min(iteration), np.max(iteration), 11)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(map(lambda x : str(int(x)), x_ticks))
        fig.canvas.draw()
        plt.pause(5)

    plt.close()

def list_columns(args):
    df = pd.read_csv(args.input);
    columns = np.setdiff1d(df.columns, ['iteration', 'log_probability', 'ensemble_index'])
    print('\n'.join(columns))

if __name__ == "__main__":

    p = create_parser()
    args = p.parse_args()

    if args.mode == "corner":
        corner_plot(args)
    elif args.mode == "plume":
        plume_plot(args)
    elif args.mode == "list":
        list_columns(args)
    elif args.mode == "progress":
        progress_plot(args)
    elif args.mode == "one_vs_all":
        one_vs_all_plot(args)
    else:
        print("Invalid mode")

