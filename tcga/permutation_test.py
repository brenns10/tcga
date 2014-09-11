#-------------------------------------------------------------------------------
#
# File:         experiment.py
#
# Author:       Stephen Brennan
#
# Date Created: Thursday, 28 August 2014
#
# Description:  Contains PermutationTest class and assorted functions.
#
#-------------------------------------------------------------------------------

import random

from .experiment import Experiment
from .tree import dag_pattern_recover, get_max_root
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
    """
    Experiment to compare pattern detection on real mutations vs random ones.

    This experiment does the following task many times: create a random
    dataset similar to the real mutations dataset.  Run the pattern detection
    procedure on the randomly generated one.  Return the mutual information
    associated with the best pattern in the DAG.

    Coupled with the mutual information, this can provide evidence that a
    real pattern has been detected.
    """

    def __init__(self, trials=1000):
        self.muts, self.phen = parse.data()
        self.dag = parse.dag()
        self.sparse = parse.sparse_mutations()
        self.results = []
        self.by_gene = {}
        self.params['Trial'] = range(trials)
        self.trials = trials

    def run_task(self, config):
        taskid = config[0]
        rand_muts = randomize_mutations(self.muts, self.sparse)
        dag_copy = self.dag.copy()
        by_gene = {}
        dag_pattern_recover(rand_muts, self.phen, dag_copy, by_gene=by_gene)
        maxroot = get_max_root(dag_copy)
        return dag_copy.node[maxroot]['mutual_info'], by_gene

    def task_callback(self, retval):
        mi, by_gene = retval
        self.results.append(mi)
        for k, v in by_gene.items():
            l = self.by_gene.get(k, [])
            l.append(v)
            self.by_gene[k] = l
