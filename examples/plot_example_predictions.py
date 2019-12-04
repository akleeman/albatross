import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

X_NAME = 'feature'
Y_NAME = 'prediction'
Y_VARIANCE_NAME = 'prediction_variance'

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("train")
    p.add_argument("predictions")
    p.add_argument("--output")
    return p

def plot_training_data(training_data, ax):
    # Show the training points
    ax.scatter(training_data[X_NAME],
                training_data['target'], color='k',
                label='training points')

def plot_truth(predictions_data, ax):
    # Plot the truth
    ax.plot(predictions_data[X_NAME],
             predictions_data['target'].astype('float'), color='black',
             label='truth')



def plot_prediction(predictions_data, model_name, color, ax):


    std = np.sqrt(predictions_data[Y_VARIANCE_NAME].values)

    # create +/- 3 sigma shading
    ax.fill_between(predictions_data[X_NAME],
                     predictions_data[Y_NAME] - 3 * std,
                     predictions_data[Y_NAME] + 3 * std, color=color, alpha=0.1,
                     label='+/- 3 sigma')
    # and +/- 1 sigma shading
    ax.fill_between(predictions_data[X_NAME],
                     predictions_data[Y_NAME] - std,
                     predictions_data[Y_NAME] + std, color=color,
                     alpha=0.5, label='+/- sigma')
    # Plot the mean
    ax.plot(predictions_data[X_NAME],
             predictions_data[Y_NAME], color=color,
             label=model_name)
    
    
def read_predictions(args):
    if ',' in args.predictions:
        model_pairs = [x.split('=') for x in args.predictions.split(',')]
        return {k: pd.read_csv(v) for k, v in model_pairs}
    else:
        return {'model': pd.read_csv(args.predictions)}

if __name__ == "__main__":

    p = create_parser()
    args = p.parse_args()

    # read in the training and prediction data
    print(args.train)
    training_data = pd.read_csv(args.train)
    predictions_data = read_predictions(args)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    colors = ['steelblue', 'firebrick', 'green']
    for (model_name, data), color in zip(predictions_data.items(), colors):
        plot_prediction(data, model_name, color, ax)
    
    plot_training_data(training_data, ax)

    example_predictions = next(iter(predictions_data.values()))
    plot_truth(example_predictions, ax)

    y_min = np.min(example_predictions['target'].astype('float'))
    y_max = np.max(example_predictions['target'].astype('float'))
    y_range = y_max - y_min
    plt.ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    fig.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
