"""Contains the DetectionExperiment class and related analysis functions."""

import os.path
import random

from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame, Series

from tcga import compare
from tcga.util import dataframe_append
from smbio.experiment import Experiment
import smbio.math.information as info


class DetectionExperiment(Experiment):
    """
    Encapsulates an experiment on how detectable functions are.

    **Task**

    The configuration task attempts to identify a pattern that has been put
    into a random dataset.  It does this self.TRIALS_PER_CONFIG times for
    each function.  It returns information about false positives and
    negatives, and about the mutual information.

    In the DetectionExperiment, a configuration is a value for:
    - Sample Size (number of items in dataset)
    - Sparsity (P(1) in each dataset)
    - Pattern Density (% of phenotype items to overwrite with pattern value)
    """

    def __init__(self, function):
        """
        Set up a detection experiment with default values.
        """
        super().__init__()
        self.function = function
        self._params['Sample Size'] = [500, 1000, 1500]
        self._params['Sparsity'] = np.arange(0.05, 0.51, 0.05)
        self._params['Pattern Density'] = np.arange(0.3, 0.0, -0.01)
        detection_columns = list(self._params.keys())
        self._params['Trials'] = [50]
        self._params['Function'] = [function]
        detection_columns += list(c.__name__ for c in compare.COMBINATIONS)
        detection_columns.append('joint')
        self.results = DataFrame(columns=detection_columns)

    def result(self, retval):
        for result in retval:
            dataframe_append(self.results, result)

    @staticmethod
    def task(configuration):
        """
        Run a single trial of the configuration task, with the given function.

        This is basically an enhanced tcga.compare.reclaim_pattern,
        which returns all the mutual information values, the reclaimed
        function, and the mutual info between the joint distribution and the
        phenotype.
        :param configuration: The experiment configuration.
        :param function: The pattern function.
        :return: Tuple:
          [0]: dict: function -> mutual information
          [1]: function: The function with the maximum mutual information.
          [2]: The mutual information etween the joint distribution and the
          phenotype.
        """
        sample_size, sparsity, pattern_density, trial, function = configuration
        results = []

        for _ in range(trial):
            res = {'Sample Size': sample_size, 'Sparsity': sparsity,
                   'Pattern Density': pattern_density}
            # Create three random datasets:
            ds1 = binary_distribution(sample_size, sparsity)
            ds2 = binary_distribution(sample_size, sparsity)
            phen = binary_distribution(sample_size, sparsity)

            # Compute f(ds1, ds2) for each function f.
            combinations = {f: f(ds1, ds2) for f in compare.COMBINATIONS}

            # Implant the pattern of function in a proportion of the phenotype.
            amount_to_implant = int(sample_size * pattern_density)
            implant_indices = random.sample(range(sample_size),
                                            amount_to_implant)
            phen[implant_indices] = combinations[function][implant_indices]

            # Calculate mutual information between function values
            phen_entropy = info.entropy(phen)
            for func, comb in combinations.items():
                res[func.__name__] = info.mutual_info_fast(comb, phen,
                                                           info.entropy(comb),
                                                           phen_entropy)

                # Calculate the mutual information of the joint distribution
                # and the phenotype.
                joint = info.joint_dataset(ds1, ds2)
                res['joint'] = info.mutual_info_fast(joint, phen,
                                                     info.entropy(joint),
                                                     phen_entropy)
            results.append(res)
        return results

    def save(self, filename='result.pickle'):
        """
        Pickle the result data frames.  Stores all files in save_dir.  They
        can be given custom filenames, or saved as 'resultN.pickle'.
        :param save_dir: Directory to save in.
        :param filenames: File names for each result DataFrame.
        """
        self.results.to_pickle(filename)


def _heatmap(subset, function_name, x_field, y_field,
             count_suffix, **kwargs):
    """
    Create a heat map.
    :param dataframe: The data to use.
    :param function_name: The function that will be plotted.
    :param x_field: The field that we will plot on the X Axis.
    :param y_field: The field that we will plot on the Y Axis.
    :param count_suffix: The filed that will specify the counts.
    :param kwargs: Keyword args to pass to hist2d().
    :return: Tuple:
     [0]: Figure
     [1]: Subplot/Axes
     [2]: 2D Histogram object.
    """
    x = []
    y = []
    for idx, row in subset.iterrows():
        count = row[function_name + count_suffix]
        x += [row[x_field]] * count
        y += [row[y_field]] * count
    fig = plt.figure()
    plot = fig.add_subplot(1, 1, 1)
    *_, hist = plot.hist2d(x, y, bins=(len(set(x)), len(set(y))), **kwargs)
    return fig, plot, hist


def plot_detection_heat_map(dataframe, sample_size, function_name):
    """
    Creates a heatmap (2D histogram) of reclaimability by sparsity.
    :param dataframe: The data produced by the DetectionExperiment.
    :param sample_size: Which sample_size to plot.
    :param function_name: The function to plot.
    :return: A matplotlib Figure.
    """
    fig, plot, hist = _heatmap(dataframe[dataframe['Sample Size'] == sample_size],
                               function_name, 'Sparsity', 'Pattern Density',
                               '_ident')
    plot.set_xlabel('Dataset Sparsity')
    plot.set_ylabel('Pattern Density')
    plot.set_title('Pattern Recovery for %s, Sample Size=%d' %
                   (function_name, sample_size))
    fig.colorbar(hist, label='Number of Times Succesfully Recovered')
    return fig


def plot_guess_heat_map(dataframe, sample_size, function_name, match_colors=True):
    """
    Creates a heatmap (2D histogram) of reclaimability by sparsity.
    :param dataframe: The data produced by the DetectionExperiment.
    :param sample_size: Which sample_size to plot.
    :param function_name: The function to plot.
    :return: A matplotlib Figure.
    """
    subset = dataframe[dataframe['Sample Size'] == sample_size]
    max_guess = min_guess = None
    if match_colors:
        max_guess = 0
        min_guess = float('inf')  # infinity
        for func in compare.COMBINATIONS:
            max_guess = max(max_guess, max(subset[func.__name__ + '_guess']))
            min_guess = min(min_guess, min(subset[func.__name__ + '_guess']))
    fig, plot, hist = _heatmap(subset, function_name,
                               'Sparsity', 'Pattern Density', '_guess',
                               vmin=min_guess, vmax=max_guess)
    plot.set_xlabel('Dataset Sparsity')
    plot.set_ylabel('Pattern Density')
    plot.set_title('Pattern Guesses for %s, Sample Size=%d' %
                   (function_name, sample_size))
    total = len(compare.COMBINATIONS) * dataframe.loc[dataframe.index[0],
                                                      function_name +
                                                      '_implant']
    fig.colorbar(hist, label='Number of Times Guessed (out of %d)' % total)
    return fig


def plot_incorrect_heat_map(dataframe, sample_size, function_name,
                            match_colors=True):
    """
    Creates a heatmap (2D histogram) of incorrect guesses.
    :param dataframe: The data produced by the DetectionExperiment.
    :param sample_size: Which sample_size to plot.
    :param function_name: The function to plot.
    :return: A matplotlib Figure.
    """
    subset = dataframe[dataframe['Sample Size'] == sample_size]
    for func in compare.COMBINATIONS:
        name = func.__name__
        subset[name + '_miss'] = subset[name + '_guess'] - subset[name +
                                                                    '_ident']

    max_guess = min_guess = None
    if match_colors:
        max_guess = 0
        min_guess = float('inf')  # infinity
        for func in compare.COMBINATIONS:
            max_guess = max(max_guess, max(subset[func.__name__ + '_miss']))
            min_guess = min(min_guess, min(subset[func.__name__ + '_miss']))
    fig, plot, hist = _heatmap(subset, function_name, 'Sparsity',
                               'Pattern Density', '_miss', vmin=min_guess,
                               vmax=max_guess)
    plot.set_xlabel('Dataset Sparsity')
    plot.set_ylabel('Pattern Density')
    plot.set_title('Incorrect Guesses for %s, Sample Size=%d' %
                   (function_name, sample_size))
    fig.colorbar(hist, label='Number of Times Incorrectly Guessed')
    return fig


def plot_all_heat_maps(dataframe, dir='.', format='svg',
                       create_func=plot_detection_heat_map):
    """
    Plots the heat maps for every sample_size and function contained in the
    dataframe.  Saves them as images in a specified directory.
    :param dataframe: The data source to plot from.
    :param dir: The directory to save in.  Accepts environment variables etc.
    :return: Nothing.
    """
    dir = os.path.expandvars(os.path.expanduser(dir))
    for sample_size in set(dataframe['Sample Size']):
        for function in compare.COMBINATIONS:
            fig = create_func(dataframe, sample_size, function.__name__)
            filename = '%d_%s.%s' % (sample_size, function.__name__, format)
            fig.savefig(os.path.join(dir, filename), format=format)


def plot_detection_comparison(dataframe, sample_size, cutoff=0.8):
    """
    Plot comparisons between each function with a given sample_size and cutoff.
    :param dataframe: Data frame to take the data from.
    :param sample_size: Sample Size to plot.
    :param cutoff: Ratio of identified/implanted that the lines should trace.
    :return: The figure produced.
    """
    fig = plt.figure()
    plot = fig.add_subplot(1, 1, 1)
    subset = dataframe[dataframe['Sample Size'] == sample_size]
    x_axis = sorted(set(dataframe['Sparsity']))
    y_axes = {f: [] for f in compare.COMBINATIONS}
    for dist in x_axis:
        dist_subset = subset[subset['Sparsity'] == dist]
        for func in compare.COMBINATIONS:
            ident = func.__name__ + '_ident'
            implant = func.__name__ + '_implant'
            props_over_cutoff = dist_subset[dist_subset[ident] / dist_subset[
                implant] >= cutoff]
            if len(props_over_cutoff) == 0:
                y_axes[func].append(None)
            else:
                y_axes[func].append(min(props_over_cutoff['Pattern Density']))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for function, color in zip(y_axes.keys(), colors):
        plot.plot(x_axis, y_axes[function], color + '-',
                  label=function.__name__)
    plot.set_xbound(lower=min(x_axis), upper=max(x_axis))
    plot.set_ybound(lower=0)
    plot.set_xlabel('Sparsity')
    plot.set_ylabel('Pattern Density')
    plot.set_title('Comparison of Function Recovery, Cutoff=%.2f, '
                   'Sample Size = %d' % (cutoff, sample_size))
    plot.legend()
    return fig


def plot_mi_comparison(dataframe, func_name, sample_size, sparsity):
    subset = dataframe[dataframe['Sample Size'] == sample_size]
    subset = subset[subset['Sparsity'] == sparsity]
    subset = subset.sort('Pattern Density')
    fig = plt.figure()
    plot = fig.add_subplot(111)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for function, color in zip(compare.COMBINATIONS, colors):
        colname = function.__name__ + '_mean'
        plot.plot(subset['Pattern Density'], subset[colname], color + '-',
                  label=function.__name__)
    plot.plot(subset['Pattern Density'], subset['joint_mean'], 'k-',
              label='joint_mean')
    plot.set_xlabel('Pattern Density')
    plot.set_ylabel('Mean Mutual Information with Phenotype (200 trials)')
    plot.set_title('Mutual Information by Pattern Density of %s for Various '
                   'Functions\nSample Size=%d, Sparsity=%.2f' %
                   (func_name, sample_size, sparsity))
    plot.legend(loc='upper left')
    return fig


def binary_distribution(size, sparsity):
    """
    Generate a boolean dataset of a particular size specified probability.

    :param size: The number of items in the dataset.
    :param sparsity: The probability of a 1/True.
    :return: A Series of bools randomly generated with the given parameters.
    """
    ds = Series(np.random.ranf(size))
    ds[ds < sparsity] = 1
    ds[ds != 1] = 0
    return ds.astype(bool)
