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
import collections

import sympy as s
from pandas import DataFrame
import networkx as nx

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


def sym_x(x, y):
    return x


def sym_y(x, y):
    return y


# This dictionary maps the dataset functions to the symbolic functions, so we
# can build expressions from the tree we run.
DS_TO_SYM = {
    compare.ds_and: sym_and,
    compare.ds_or: sym_or,
    compare.ds_xor: sym_xor,
    compare.ds_and_not: sym_and_not,
    compare.ds_not_and: sym_not_and,
    compare.ds_x: sym_x,
    compare.ds_y: sym_y,
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


def dag_pattern_recover(muts, phen, dag):
    leaves = (n for n, d in dag.out_degree_iter() if d == 0)
    nodes = collections.deque()

    nx.set_node_attributes(dag, 'visited', False)
    nx.set_node_attributes(dag, 'dataset', None)
    nx.set_node_attributes(dag, 'function', None)
    nx.set_node_attributes(dag, 'mutual_info', None)

    print('Examining leaf nodes ...')
    # Look at all leaf nodes, and make sure that they are contained in the
    # mutation dataset.
    for leaf in leaves:
        # Set up the dataset parameter.
        params = dag.node[leaf]
        if leaf not in muts.columns:
            print('=> No mutation data for leaf node %s, ignoring.' % str(leaf))
            params['dataset'] = None
        else:
            params['dataset'] = leaf
        params['visited'] = True

        # Add its parent, if all its children are leaves
        for pred in dag.predecessors(leaf):
            if all(dag.node[succ]['visited'] for succ in dag.successors(pred)):
                nodes.appendleft(pred)

    print('Finding best combinations ...')
    while nodes:
        # Get the current node and mark it as visited.
        curr = nodes.pop()
        params = dag.node[curr]
        params['visited'] = True

        # Get the best combination of its children.
        children = dag.successors(curr)
        if len(children) != 2:
            raise Exception('Invalid degree of node ' + str(curr))
        x_key = dag.node[children[0]]['dataset']
        y_key = dag.node[children[1]]['dataset']
        if x_key is None:
            if y_key is None:
                print('=> No mutation for either child of node %s.' % str(curr))
                params['dataset'] = None
            else:
                params['dataset'] = y_key
                params['function'] = compare.ds_y
                params['mutual_info'] = dag.node[children[1]]['mutual_info']
        else:
            if y_key is None:
                params['dataset'] = x_key
                params['function'] = compare.ds_x
                params['mutual_info'] = dag.node[children[0]]['mutual_info']
            else:
                function, dataset, mi, *etc = compare.best_combination(
                    muts[x_key], muts[y_key], phen)
                params['function'] = function
                params['dataset'] = curr
                muts[curr] = dataset
                params['mutual_info'] = mi

        for pred in dag.predecessors(curr):
            if all(dag.node[succ]['visited'] for succ in dag.successors(pred)):
                nodes.appendleft(pred)

    print('Done!')


def get_roots(dag):
    return (x for x, d in dag.in_degree_iter() if d == 0)


def get_max_root(dag):
    maxmi = 0
    maxroot = None
    for root in get_roots(dag):
        mi = dag.node[root]['mutual_info']
        if mi is not None and mi > maxmi:
            maxmi = mi
            maxroot = root
    return maxroot


def _get_subtree(dag, node, curr):
    curr.append(node)
    func = dag.node[node]
    successors = dag.successors(node)
    if successors:
        if func == compare.ds_x:
            _get_subtree(dag, successors[0], curr)
        elif func == compare.ds_y:
            _get_subtree(dag, successors[1], curr)
        else:
            _get_subtree(dag, successors[0], curr)
            _get_subtree(dag, successors[1], curr)


def get_function_subtree(dag, node):
    nodes = []
    _get_subtree(dag, node, nodes)
    return dag.subgraph(nodes)


def get_function(dag, node):
    f = dag.node[node]['function']
    if f is None:
        return s.Symbol('[' + str(node) + ']')
    else:
        succ = dag.successors(node)
        return DS_TO_SYM[f](get_function(dag, succ[0]), get_function(dag, succ[1]))


def count_disconnected(dag):
    count = 0
    nx.set_node_attributes(dag, 'group', None)
    for node in dag.nodes_iter():
        if dag.node[node]['group'] is None:
            count += 1
            q = collections.deque()
            q.appendleft(node)
            while q:
                curr = q.pop()
                if dag.node[curr]['group'] is None:
                    dag.node[curr]['group'] = count
                    for succ in dag.neighbors(curr):
                        q.appendleft(succ)
    return count


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