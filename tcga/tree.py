#-------------------------------------------------------------------------------
# 
# File:         tree.py
#
# Author:       Stephen Brennan
#
# Date Created: Wednesday,  4 June 2014
#
# Description:  Chooses best binary functions on trees.
#
#-------------------------------------------------------------------------------

import random

import sympy as s
from pandas import DataFrame

from . import compare, parse, util


# These are functions that take expressions and combine them using whatever
# symbol they are named for.
def sym_and(x, y):
    return x & y


def sym_or(x, y):
    return x | y


def sym_xor(x, y):
    return x ^ y


def sym_and_not(x, y):
    return x & ~y


def sym_not_and(x, y):
    return ~x & y


# This dictionary maps the dataset functions to the symbolic functions, so we
# can build expressions from the tree we run.
DS_TO_SYM = {
    compare.ds_and: sym_and,
    compare.ds_or: sym_or,
    compare.ds_xor: sym_xor,
    compare.ds_and_not: sym_and_not,
    compare.ds_not_and: sym_not_and,
}


def random_tree(muts, phen, depth, verbose=True, simplify=False):
    """Create a boolean expression for a random tree of genes.

    For the given mutation and phenotype data, construct a random tree/ontology
    of genes, and find the boolean expression that best relates the mutations to
    the phenotype.  Returns the SymPy expression generated.

    muts - Mutation data in a pandas DataFrame
    phen - Phenotype data in a pandas Series
    depth - The depth of the tree.  Must be >=1
    verbose - Print info about what is happening
    simplify - Do we simplify the expression before returning (can be lengthy)
    """
    # Create a random sample from the entire list of possible genes
    if verbose:
        print("Selecting gene sample ...")
    choices = iter(random.sample(range(len(muts.columns)), depth + 1))

    # Get two genes from the sample (as string names).
    firststr = muts.columns[next(choices)]
    secondstr = muts.columns[next(choices)]

    # Create SymPy symbols from the gene names
    firstsym = s.Symbol('[' + firststr + ']')
    secondsym = s.Symbol('[' + secondstr + ']')

    # Get the actual Series for each gene
    first = muts[firststr]
    second = muts[secondstr]

    # Choose our first combination
    print("Choosing between %s and %s." % (firststr, secondstr))
    func, prev, mi1, mi2 = compare.best_combination(first, second, phen)
    print("Selecting %s, MI=%f, by margin of %f." % (
        func.__name__, mi1, mi1 - mi2))

    # Construct an expression using the function returned
    expr = DS_TO_SYM[func](firstsym, secondsym)

    # Now, begin the loop for the rest of the random tree.  In each iteration,
    # get the next gene from the random sample, find the best combination with
    # the existing dataset function, and add it to the result expression.
    for curr in choices:
        currstr = muts.columns[curr]
        print("=> Adding %s." % currstr)
        currdataset = muts[currstr]
        func, prev, mi1, mi2 = compare.best_combination(prev, currdataset, phen)
        print("   Selecting %s, MI=%f, by margin of %f." % (
            func.__name__, mi1, mi1 - mi2))
        expr = DS_TO_SYM[func](expr, s.Symbol('[' + currstr + ']'))

    if simplify:
        expr = s.simplify_logic(expr, 'dnf')
    if verbose:
        print('Expression constructed:')
        s.pprint(expr)
    return expr


@util.progress_bar(size_param=(2, 'niters', None))
def rand_rectangle(mutations, sparse_mutations, niters=None):
    """
    An iterator that yields swappable rectangles from the mutations matrix.

    In order to create a randomized matrix that retains the properties of the
    original, we take the original matrix, and randomly choose two points
    such that both are True, and the opposite corners of the rectangle they
    form are False.  Graphically:

    [ 1 .... 0 ]
    [ .      . ]
    [ .      . ]
    [ 0 .... 1 ]

    This function is an iterator that yields a certain number of these
    rectangles, randomly selected.  Note that the parameters mutations and
    sparse_mutations should be updated on every swap.  Thus, it isn't
    advisable to do list(rand_rectangle(...)), as some of the later
    rectangles might be affected by previous swaps.

    :param mutations: The mutations DataFrame that we are randomizing.
    :param sparse_mutations: A sparse representation of the mutations we are
    randomizing, so we can quickly choose values which are True.
    :param niters: The number of rectangles to yield.
    :return: Tuple: (gene_1, patient_1, gene_2, patient_2).
    """
    import random

    if niters is None:
        niters = 4 * len(sparse_mutations)
    yielded = 0
    while yielded < niters:
        idx1 = random.randrange(len(sparse_mutations))
        idx2 = random.randrange(len(sparse_mutations))
        gene_1, patient_1 = sparse_mutations[idx1]
        gene_2, patient_2 = sparse_mutations[idx2]
        if not (mutations[gene_1][patient_2] or mutations[gene_2][

                patient_1]):

            yield (idx1, idx2)
            yielded += 1


def randomize_mutations(mutations, mutationsloc=parse.MUTSLOC,
                        num_iterations=None, phenotype=None):
    """
    Perform random swaps on mutation data until it is randomized.

    :param mutations: Mutations dataset.
    :param mutationsloc: location of the mutations CSV
    :param num_iterations: Number of swaps to make.
    :param phenotype: Phenotype dataset.
    :return: Randomized dataset.
    """
    new_mutations = mutations.copy()
    sparse_mutations = parse.sparse_mutations(filename=mutationsloc,
                                              phenotype=phenotype)
    if num_iterations is None:
        num_iterations = 4 * len(sparse_mutations)

    for idx1, idx2 in rand_rectangle(new_mutations, sparse_mutations,
                                     num_iterations):
        gene_1, patient_1 = sparse_mutations[idx1]
        gene_2, patient_2 = sparse_mutations[idx2]
        new_mutations[gene_1][patient_2] = True
        new_mutations[gene_2][patient_1] = True
        new_mutations[gene_1][patient_1] = False
        new_mutations[gene_2][patient_2] = False
        del sparse_mutations[idx1]
        del sparse_mutations[idx2 - 1]
        sparse_mutations.append((gene_1, patient_2))
        sparse_mutations.append((gene_2, patient_1))
    return new_mutations


def greedy_tree(muts, phen, copy=True, out=True):
    """
    Greedily construct a tree from the mutations dataset.

    Treats the mutation dataset as a pool of root nodes to connect.  Chooses
    the pair of nodes with the highest mutual info and combines them,
    creating a new node and removing those two.

    :param muts: The mutations dataset.
    :param phen: The phenotype dataset.
    :param copy: True if we copy muts before modifying.
    :return:
    """
    if copy:
        if out:
            print('Copying mutation data...')
        muts = muts.copy()

    # Create gene x gene matrices to store mutual info and functions.  The
    # matrices should be triangular, like this:
    #
    #      0   1   2   3
    # 0  [ na  na  na  na ]
    # 1  [ xx  na  na  na ]
    # 2  [ xx  xx  na  na ]
    # 3  [ xx  xx  xx  na ]
    #
    if out:
        print('Creating storage matrices...')
    mutual_info = DataFrame(index=muts.columns, columns=muts.columns,
                            dtype=float)
    best_function = DataFrame(index=muts.columns, columns=muts.columns,
                              dtype='object')

    # Populate matrices
    if out:
        print('Populating matrices...')
    for gene1idx in range(len(mutual_info.index)):  # iterate by row
        gene1 = mutual_info.index[gene1idx]

        for gene2idx in range(gene1idx):  # column
            gene2 = mutual_info.columns[gene2idx]

            func, dataset, info, *etc = compare.best_combination(muts[gene1],
                                                                 muts[gene2],
                                                                 phen)
            mutual_info[gene2][gene1] = info
            best_function[gene2][gene1] = func
            if out:
                print('(%s, %s): %s (%f)' % (gene1, gene2, func.__name__, info))

    # Repeat until we have only one (root) node:

    # Retrieve max.
    max_mi = mutual_info.max(axis=0).max()
    if out:
        print(max_mi)

    # Create a new node with the two max-MI nodes.
    # Remove the two nodes from the data set.
    # Calculate mutual information between new node and the others, add to set.

    # Finally, output tree.