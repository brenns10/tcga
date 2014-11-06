"""Contains the Experiment class."""

import itertools as it
import multiprocessing as mp
import traceback
from abc import ABCMeta, abstractmethod
from collections import OrderedDict


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
