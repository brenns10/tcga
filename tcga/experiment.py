#-------------------------------------------------------------------------------
#
# File:         compare.py
#
# Author:       Stephen Brennan
#
# Date Created: Wednesday,  4 June 2014
#
# Description:  Contains functions to compare binary data sets.
#
#-------------------------------------------------------------------------------

import itertools as it
import multiprocessing as mp
import random
import os.path
import traceback

import numpy as np
from pandas import DataFrame

from . import compare


def _dataframe_append(dataframe, rowdict):
    newrow = len(dataframe)
    dataframe.loc[newrow] = 0  # init with 0's
    for k, v in rowdict.items():
        dataframe.loc[newrow, k] = v


class DetectionExperiment:
    """
    This class encapsulates an experiment to learn more about the detection
    of various patterns implanted at various proportions into datasets of
    various distributions.  This can be done at various 'degrees' (that is,
    lengths of datasets).  In other words, there are many possible
    combinations of parameters this experiment could run at.  This class
    helps eliminate deep nested for-loops and long function chains that make
    custimizing parameter ranges difficult or impossible.

    It divides the experiment into 'configurations'.  A single configuration
    is a single set of possible parameter values.  A configuration has a task
    that should be run with those parameters (self.run_config()).
    Configuration tasks are run with parameter values and return rows to be
    added to the result DataFrame(s).

    In the DetectionExperiment, a configuration is a value for:
    - Degree (number of items in dataset)
    - Distribution (P(1) in each dataset)
    - Proportion (% of phenotype items to overwrite with pattern value)
    The configuration task attempts to reclaim an imprinted pattern
    self.TRIALS_PER_CONFIG times for each function.  It returns information
    about false positives and negatives, and about the mutual information.

    Configurations are runnable in parallel with the multiprocessing module.  I
    may add a serial running option (for some reason).  I may also spin out
    the machinery behind the basic experiment framework into a parent class.
    """

    def __init__(self):
        """
        Set up a detection experiment with default values.
        """
        self.degrees = [500, 1000, 1500]
        self.distributions = list(np.arange(0.05, 0.51, 0.05))
        self.proportions = list(np.arange(0.3, 0.0, -0.01))
        self.TRIALS_PER_CONFIG = 50
        self.configuration_columns = ['Degree', 'Distribution', 'Proportion']

        # Define the column for the detection dataset.
        detection_columns = list(self.configuration_columns)
        for function in compare.COMBINATIONS:
            detection_columns.append(function.__name__ + '_implant')
            detection_columns.append(function.__name__ + '_ident')
            detection_columns.append(function.__name__ + '_guess')

        self.detection_results = DataFrame(columns=detection_columns)

        # Define the columns for the mutual information dataset.
        mi_columns = list(self.configuration_columns)
        mi_columns.append('Implanted')
        mi_columns.append('Trials')
        mi_columns.append('joint_mean')
        mi_columns.append('joint_std')
        for function in compare.COMBINATIONS:
            mi_columns.append(function.__name__ + '_mean')
            mi_columns.append(function.__name__ + '_std')

        self.mutual_info_results = DataFrame(columns=mi_columns)

    def run_trial(self, configuration, function):
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
        degree, distribution, proportion = configuration

        # Create three random datasets:
        ds1 = compare.binary_distribution(degree, distribution)
        ds2 = compare.binary_distribution(degree, distribution)
        phen = compare.binary_distribution(degree, distribution)

        # Compute f(ds1, ds2) for each function f.
        combinations = {f: f(ds1, ds2) for f in compare.COMBINATIONS}

        # Implant the pattern of function in a proportion of the phenotype.
        amount_to_implant = int(degree * proportion)
        for i in random.sample(range(degree), amount_to_implant):
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

    def run_config(self, configuration):
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
                'Degree': configuration[0],
                'Distribution': configuration[1],
                'Proportion': configuration[2],
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
            'Degree': configuration[0],
            'Distribution': configuration[1],
            'Proportion': configuration[2]
        }
        for func in compare.COMBINATIONS:
            detection_row[func.__name__ + '_implant'] = implant[func]
            detection_row[func.__name__ + '_ident'] = ident[func]
            detection_row[func.__name__ + '_guess'] = guess[func]

        # Return our detection row and mutual information rows.
        return detection_row, mi_rows

    def run_config_wrapped(self, configuration):
        """
        Provides decent error handling in the multiprocessing module.

        Since the multiprocessing module doesn't really allow you to get
        things like stack traces from exceptions, this function wraps the
        run_config() function, and at catches all exceptions, adding a
        traceback in text form, so that error callback function can display
        the traceback.
        :param configuration: Passed to run_config()
        :return: Return from run_config()
        """
        try:
            return self.run_config(configuration)
        except Exception:
            raise Exception("".join(traceback.format_exc()))

    def experiment_callback(self, retVal):
        """
        Store data produced by run_config().

        This function is called by the multiprocessing module when a task is
        completed.  It is executed on the main process, so no synchronization is
        needed to access the DataFrame.
        :param retVal: Value returned by run_config().
        :return:
        """
        detection_row, mi_rows = retVal
        _dataframe_append(self.detection_results, detection_row)
        for row in mi_rows:
            _dataframe_append(self.mutual_info_results, row)
        self.completed += 1
        print('Completed %d of %d.' % (self.completed, self.nconfigs))

    def errcb(self, exception):
        """
        Error callback for multiprocessing.

        This function 'handles' exceptions from Processes by displaying them.
        The run_config_wrapped() function puts tracebacks into the
        exceptions, so that printing them here is actually meaningful.
        :param exception: The exception thrown by run_config()
        :return:
        """
        print(exception)

    def run_experiment_multiprocessing(self, processes=None):
        """
        Runs the experiment using the multiprocessing module.

        A process worker pool is created, and tasks are delegated to each
        worker.  Since there is some process spawning overhead, as well as
        IPC overhead, this isn't perfect.  Tasks should be slow enough that
        the speed gains of parallelizing outweigh the overhead of spawning
        and IPC.
        :param processes: Number of processes to use in the pool.  Default is
        None. If None is given, the number from multiprocessing.cpu_count()
        is used.
        :return: Blocks until all tasks are complete.  Returns nothing.
        """
        configs = list(it.product(self.degrees, self.distributions,
                                  self.proportions))
        self.nconfigs = len(configs)
        self.completed = 0
        resultobjs = []

        print('Running %d configurations.' % self.nconfigs)

        with mp.Pool(processes=processes) as pool:
            for configuration in configs:
                resultobjs.append(pool.apply_async(self.run_config_wrapped,
                    (configuration,), callback=self.experiment_callback,
                    error_callback=self.errcb))
            print('All jobs submitted.')
            for result in resultobjs:
                result.wait()

    def save(self, savedir, detection_results_name='detection_results.pickle',
             mutual_info_results_name='mutual_info_results.pickle'):
        """
        Picle the result data frames.
        :param savedir: Directory to save in.
        :param detection_results_name: Name for the detection results.
        :param mutual_info_results_name: Name for the mutual info results.
        """
        savedir = os.path.expandvars(os.path.expanduser(savedir))
        self.detection_results.to_pickle(os.path.join(savedir,
                                                      detection_results_name))
        self.mutual_info_results.to_pickle(os.path.join(savedir,
                                                        mutual_info_results_name))