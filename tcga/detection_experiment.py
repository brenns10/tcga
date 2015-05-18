"""Contains the DetectionExperiment class and related analysis functions."""

import itertools as it
import os.path
import random

from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame, Series

from . import compare
from .experiment import Experiment


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

    def __init__(self):
        """
        Set up a detection experiment with default values.
        """
        super().__init__()
        self._params['Sample Size'] = [500, 1000, 1500]
        self._params['Sparsity'] = np.arange(0.05, 0.51, 0.05)
        self._params['Pattern Density'] = np.arange(0.3, 0.0, -0.01)
        self.TRIALS_PER_CONFIG = 50
        self.results = []

        # Define the column for the detection dataset.
        detection_columns = list(self._params.keys())
        for function in compare.COMBINATIONS:
            detection_columns.append(function.__name__ + '_implant')
            detection_columns.append(function.__name__ + '_ident')
            detection_columns.append(function.__name__ + '_guess')

        self.results.append(DataFrame(columns=detection_columns))

        # Define the columns for the mutual information dataset.
        mi_columns = list(self._params.keys())
        mi_columns.append('Implanted')
        mi_columns.append('Trials')
        mi_columns.append('joint_mean')
        mi_columns.append('joint_std')
        for function in compare.COMBINATIONS:
            mi_columns.append(function.__name__ + '_mean')
            mi_columns.append(function.__name__ + '_std')

        self.results.append(DataFrame(columns=mi_columns))

    def result(self, retval):
        for row_list, dataframe in zip(retval, self.results):
            for row in row_list:
                self._dataframe_append(dataframe, row)

    @staticmethod
    def trial(configuration, function):
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
        sample_size, sparsity, pattern_density = configuration

        # Create three random datasets:
        ds1 = binary_distribution(sample_size, sparsity)
        ds2 = binary_distribution(sample_size, sparsity)
        phen = binary_distribution(sample_size, sparsity)

        # Compute f(ds1, ds2) for each function f.
        combinations = {f: f(ds1, ds2) for f in compare.COMBINATIONS}

        # Implant the pattern of function in a proportion of the phenotype.
        amount_to_implant = int(sample_size * pattern_density)
        for i in random.sample(range(sample_size), amount_to_implant):
            phen[i] = combinations[function][i]

        # Calculate mutual information between function values
        mutual_info = {}
        for func, comb in combinations.items():
            mutual_info[func] = compare.mutual_info(comb, phen)

        # Select the best function based on mutual info.
        max_f = max(mutual_info.keys(), key=lambda k: mutual_info[k])

        # Calculate the mutual information of the joint distribution and the
        # phenotype.
        joint_distribution = ds1 + 2*ds2
        joint_mutual_info = compare.mutual_info(joint_distribution, phen,
                                                ds1domain=4)
        return mutual_info, max_f, joint_mutual_info

    def task(self, configuration):
        """
        Run a single configuration.
        :param configuration: The configuration parameters.
        :return: Tuple:
         [0]: dict: Row to add to the detection dataset.
         [1]: list of dict: Rows to add to the mutual information dataset.
        """
        # Setup result dictionaries
        implant = {f: 0 for f in compare.COMBINATIONS}
        ident = {f: 0 for f in compare.COMBINATIONS}
        guess = {f: 0 for f in compare.COMBINATIONS}
        mi_rows = []

        for func in compare.COMBINATIONS:
            mutual_infos = {f: [] for f in compare.COMBINATIONS}
            joint_mis = []

            for _ in range(self.TRIALS_PER_CONFIG):
                # Run the trial
                midict, res_func, joint_mi = self.run_trial(configuration, func)

                # Record stats about recovery
                implant[func] += 1
                if func == res_func:
                    ident[func] += 1
                guess[res_func] += 1

                # Record the mutual information
                for f, mi in midict.items():
                    mutual_infos[f].append(mi)
                joint_mis.append(joint_mi)

            # Generate the mutual information row for this configuration and
            # function.
            mi_row = {
                'Sample Size': configuration[0],
                'Sparsity': configuration[1],
                'Pattern Density': configuration[2],
                'Implanted': func.__name__,
                'Trials': self.TRIALS_PER_CONFIG,
                'joint_mean': np.mean(joint_mis),
                'joint_std': np.std(joint_mis),
            }
            for f, milist in midict.items():
                mi_row[f.__name__ + '_mean'] = np.mean(milist)
                mi_row[f.__name__ + '_std'] = np.std(milist)

            # Add the row to the list of mutual information rows to return.
            mi_rows.append(mi_row)

        # Generate the detection data row for this configuration.
        detection_row = {
            'Sample Size': configuration[0],
            'Sparsity': configuration[1],
            'Pattern Density': configuration[2]
        }
        for func in compare.COMBINATIONS:
            detection_row[func.__name__ + '_implant'] = implant[func]
            detection_row[func.__name__ + '_ident'] = ident[func]
            detection_row[func.__name__ + '_guess'] = guess[func]

        # Return our detection row and mutual information rows.
        return [detection_row], mi_rows

    def save(self, save_dir='.', filenames=None):
        """
        Pickle the result data frames.  Stores all files in save_dir.  They
        can be given custom filenames, or saved as 'resultN.pickle'.
        :param save_dir: Directory to save in.
        :param filenames: File names for each result DataFrame.
        """
        if filenames is None:
            filenames = ['result%d.pickle' % i for i in range(len(self.results))]

        save_dir = os.path.expandvars(os.path.expanduser(save_dir))
        for filename, result in zip(filenames, self.results):
            result.to_pickle(os.path.join(save_dir, filename))


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
    subset = subset[subset['Implanted'] == func_name]
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


def plot_all_mi_comparisons(dataframe, dir='.', format='svg'):
    fnames = [f.__name__ for f in compare.COMBINATIONS]
    ssizes = set(dataframe['Sample Size'])
    sparsities = set(dataframe['Sparsity'])
    for fname, ssize, sparsity in it.product(fnames, ssizes, sparsities):
        fig = plot_mi_comparison(dataframe, fname, ssize, sparsity)
        filename = '%d_%.2f_%s.%s' % (ssize, sparsity, fname, format)
        fig.savefig(os.path.join(dir, filename), format=format)


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