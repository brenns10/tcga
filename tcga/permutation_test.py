"""Contains PermutationTest class and assorted functions.

The PermutationTest is a subclass of Experiment, which means it can be
multiprocessed very well.  The PermutationTest randomizes the mutation
dataset, and then records the results of dag_pattern_recover() using this
mutation dataset.  Its purpose is to establish whether the result of
running dag_pattern_recover() on the real mutation dataset is significant
or not.
"""

import random

from .experiment import Experiment
from tcga import compare
from tcga.pattern import dag_pattern_recover
from .tree import terminal_function_nodes
from .compare import mutual_info
from . import parse


def randomize_mutations(mutations, sparse_mutations, num_iterations=None):
    """
    Perform random swaps on mutation data until it is randomized.

    In order to randomize the mutation data while maintaining the
    characteristics of the dataset (i.e. row & column counts), we randomly
    choose two locations where the data is True, and if the opposite corners
    of the rectangle formed by them are both False, we flip all the corners.
    Graphically:

    [ 1 .... 0 ]         [ 0 .... 1 ]
    [ .      . ] becomes [ .      . ]
    [ .      . ]         [ .      . ]
    [ 0 .... 1 ]         [ 1 .... 0 ]

    This process is repeated many times (by default, 4x the number of True
    values in the matrix, to allow each True value to have the chance to have
    flipped to False and back).

    :param mutations: Mutations dataset.  This dataset is NOT modified.
    :param sparse_mutations: Sparse mutations list.  Also NOT modified.
    :param num_iterations: Number of swaps to make.  By default,
    4*len(sparse_mutations).
    :return: A new, randomized dataset.
    """
    mutations = mutations.copy()
    sparse = sparse_mutations.copy()
    if num_iterations is None:
        num_iterations = 4 * len(sparse)

    while num_iterations:
        # Randomly select two mutations.
        idx1 = random.randrange(len(sparse))
        idx2 = random.randrange(len(sparse))
        gene1, patient1 = sparse[idx1]
        gene2, patient2 = sparse[idx2]

        # Make sure we don't swap the same gene/patient and count it
        if gene1 == gene2 and patient1 == patient2:
            continue

        # Make sure that the corners are both False
        if mutations[gene1][patient2] or mutations[gene2][patient1]:
            continue

        # Perform the swap in the full data structure.
        mutations[gene1][patient2] = True
        mutations[gene2][patient1] = True
        mutations[gene1][patient1] = False
        mutations[gene2][patient2] = False

        # Perform the swap in the sparse data structure.
        del sparse[max(idx1, idx2)]
        del sparse[min(idx1, idx2)]
        sparse.append((gene1, patient2))
        sparse.append((gene2, patient1))

        # Loop down to zero.
        num_iterations -= 1

    return mutations


class PermutationTest(Experiment):
    """Experiment to compare pattern detection on real mutations vs random ones.

    This experiment does the following task many times: create a random dataset
    similar to the real mutations dataset.  Run the pattern detection procedure
    on the randomly generated one.  Return the comparison value of the top N
    results.

    Coupled with the mutual information, this can provide evidence that a
    real pattern has been detected.

    """

    def __init__(self, phenotype='vital-status', mutations='mutations',
                 comparison=mutual_info, trials=1000, ranks=200):
        """
        Create an instance of PermutationTest.
        :param phenotype: Name of phenotype to use.
        :param mutations: Name of mutations dataset to use.
        :param comparison: Comparison to use.  Compatible with phenotype.
        :param trials: Number of times to run the permutation test.
        :param ranks: Number of top ranked values to return.
        :return: New instance of LifetimePermutationTest.
        """
        # Load necessary data:
        self.muts, self.phen = parse.data(phenotype_title=phenotype,
                                          mutation_title=mutations)
        self.sparse = parse.sparse_mutations()
        self.sparse = parse.restrict_sparse_mutations(self.sparse, self.phen,
                                                      self.muts)
        self.dag = parse.dag()

        self.ranks = ranks

        # Result list for each mutual information.
        self.results = [[] for _ in range(ranks)]
        # The comparison function to use!
        self.comparison = compare.log_rank

        # This experiment is simply repeated for many trials.  No parameters
        # are varying.
        self.params['Trial'] = range(trials)
        self.trials = trials

    def run_task(self, config):
        """
        Task that is repeated for each trial.

        Generate a random mutation dataset.  Run the dag_pattern_recover()
        function using the random dataset, along with the original phenotype
        and DAG.
        :param config: Variable containing the parameters (in this case,
        just the trial number).
        :return: A list of 'self.rank' top function values.
        """
        rand_muts = randomize_mutations(self.muts, self.sparse)
        dag_copy = self.dag.copy()
        dag_pattern_recover(rand_muts, self.phen, dag_copy,
                            comparison=self.comparison)

        nodes = list(terminal_function_nodes(dag_copy))
        values = [0] * len(nodes)
        for i, node in enumerate(nodes):
            values[i] = dag_copy.node[node]['value']
        values.sort(reverse=True)
        return values[:self.ranks]

    def task_callback(self, retval):
        """
        Store the data from the task.  Executed on main process.
        :param retval: Return value from run_task().
        :return: None
        """
        for lst, value in zip(self.results, retval):
            lst.append(value)
