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
import time
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
from pandas import DataFrame

from . import compare
from .util import Progress


class Experiment:
    """
    Abstract Base Class for experiment execution.

    This ABC encapsulates an 'experiment', which here is a task that
    generates data and is repeated for a large number possibilites for its
    parameters.  This class allows executing tasks with arbitrary numbers of
    parameters, and arbitrary value ranges for each parameter.

    This class allows these tasks to be easily executed in parallel, which can
    save time for long-running, CPU intensive tasks.  The tasks may also be
    executed in serial.
    """
    __metaclass__ = ABCMeta

    params = OrderedDict()
    completed = 0
    num_configs = 0
    progress = None

    @staticmethod
    def _dataframe_append(dataframe, rowdict):
        """
        Shortcut method for appending a row to a DataFrame.
        :param dataframe: The DataFrame to append to.
        :param rowdict: A dictionary containing each column's value.
        """
        newrow = len(dataframe)
        dataframe.loc[newrow] = 0  # init with 0's
        for k, v in rowdict.items():
            dataframe.loc[newrow, k] = v

    @staticmethod
    def error_callback(exception):
        """
        Error callback for multiprocessing.

        This function 'handles' exceptions from Processes by displaying them.
        The run_config_wrapped() function puts tracebacks into the
        exceptions, so that printing them here is actually meaningful.
        :param exception: The exception thrown by run_config()
        """
        print(exception)

    @abstractmethod
    def run_task(self, configuration):
        """
        This function must be the task that is run for every configuration.

        This function MUST be overridden.  It should take the configuration
        as its argument, and return whatever data will be saved.  (Its return
        value will be passed to task_callback()).
        :param configuration: The task configuration.
        :return: The data to be saved.
        """
        pass

    @abstractmethod
    def task_callback(self, retval):
        """
        This function should save the data created by each task.

        This function MUST be overridden.  It will be called with the return
        value of run_task().  It should save this data in a structure (like a
        DataFrame) that was initialized in __init__().
        :param retval: The value returned by run_task().
        :return: Nothing
        """
        pass

    def initial_task_callback(self, retval):
        """
        Receives callbacks from multiprocessing.

        This function is called by the multiprocessing module when a task is
        completed.  It then calls the task callback provided by overriding
        functions.
        :param retval: Value returned by run_config().
        :return:
        """
        self.completed += 1
        print('Completed %d/%d.' % (self.completed, self.num_configs))
        self.task_callback(retval)

    def run_task_wrapped(self, configuration):
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
            return self.run_task(configuration)
        except Exception:
            raise Exception("".join(traceback.format_exc()))

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
        # Setup the class variables used during the experiment.
        self.completed = 0

        # Create all possible configurations (an iterator).
        configs = it.product(*self.params.values())

        # Create a multiprocessing pool and add each configuration task.
        resultobjs = []
        with mp.Pool(processes=processes) as pool:
            for configuration in configs:
                resultobjs.append(pool.apply_async(self.run_task_wrapped,
                    (configuration,), callback=self.initial_task_callback,
                    error_callback=self.error_callback))
                self.num_configs += 1
            print('Experiment: queued %d tasks.' % self.num_configs)
            for result in resultobjs:
                result.wait()
            print('Experiment: completed all tasks.')


class DetectionExperiment(Experiment):
    """
    Encapsulates an experiment on how detectable functions are.

    In the DetectionExperiment, a configuration is a value for:
    - Degree (number of items in dataset)
    - Distribution (P(1) in each dataset)
    - Proportion (% of phenotype items to overwrite with pattern value)
    The configuration task attempts to reclaim an imprinted pattern
    self.TRIALS_PER_CONFIG times for each function.  It returns information
    about false positives and negatives, and about the mutual information.
    """

    def __init__(self):
        """
        Set up a detection experiment with default values.
        """
        self.params['Degree'] = [500, 1000, 1500]
        self.params['Distribution'] = np.arange(0.05, 0.51, 0.05)
        self.params['Proportions'] = np.arange(0.3, 0.0, -0.01)
        self.TRIALS_PER_CONFIG = 50
        self.results = []

        # Define the column for the detection dataset.
        detection_columns = list(self.params.keys())
        for function in compare.COMBINATIONS:
            detection_columns.append(function.__name__ + '_implant')
            detection_columns.append(function.__name__ + '_ident')
            detection_columns.append(function.__name__ + '_guess')

        self.results.append(DataFrame(columns=detection_columns))

        # Define the columns for the mutual information dataset.
        mi_columns = list(self.params.keys())
        mi_columns.append('Implanted')
        mi_columns.append('Trials')
        mi_columns.append('joint_mean')
        mi_columns.append('joint_std')
        for function in compare.COMBINATIONS:
            mi_columns.append(function.__name__ + '_mean')
            mi_columns.append(function.__name__ + '_std')

        self.results.append(DataFrame(columns=mi_columns))

    def task_callback(self, retval):
        for row_list, dataframe in zip(retval, self.results):
            for row in row_list:
                self._dataframe_append(dataframe, row)

    @staticmethod
    def run_trial(configuration, function):
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

    def run_task(self, configuration):
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
        return [detection_row], mi_rows

    def save(self, savedir='.', filenames=None):
        """
        Picle the result data frames.  Stores all files in savedir.  They can be
        given custom filenames, or saved as 'resultN.pickle'.  
        :param savedir: Directory to save in.
        :param filenames: File names for each result DataFrame.
        """
        if filenames is None:
            filenames = ['result%d.pickle' % i for i in range(len(self.results))]

        savedir = os.path.expandvars(os.path.expanduser(savedir))
        for filename, result in zip(filenames, self.results):
            result.to_pickle(os.path.join(savedir, filename))