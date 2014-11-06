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

import networkx as nx

import sympy as s
from . import compare


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


def get_roots(dag):
    """
    Returns the roots of a forest of DAGs.
    :param dag: A NetworkX directed graph.
    :return: A generator that yields each root node in the DAG.
    """
    return (x for x, d in dag.in_degree_iter() if d == 0)


def get_max_root(dag):
    """
    Return the DAG root that has the highest mutual information associated
    with it (after calling dag_pattern_reover() on the DAG).
    :param dag: The NetworkX DAG, which has already had dag_pattern_recover()
    called on it.
    :return: The root with the highest mutual information.
    """
    maxmi = 0
    maxroot = None
    for root in get_roots(dag):
        mi = dag.node[root]['value']
        if mi is not None and mi > maxmi:
            maxmi = mi
            maxroot = root
    return maxroot


def _get_subtree(dag, node, subtree):
    """
    Recursive helper function to get_function_subtree().  Adds the current
    node to the subtree list.  Calls the function on each child if it is part of
    the function.
    :param dag: The DAG we are finding the subtree of.
    :param node: The current node we are recursively operating on.
    :param subtree: The list of nodes in this subtree.
    :return: None
    """
    subtree.append(node)
    func = dag.node[node]
    successors = dag.successors(node)
    if successors:
        if func == compare.ds_x:
            _get_subtree(dag, successors[0], subtree)
        elif func == compare.ds_y:
            _get_subtree(dag, successors[1], subtree)
        else:
            _get_subtree(dag, successors[0], subtree)
            _get_subtree(dag, successors[1], subtree)


def get_function_subtree(dag, node):
    """
    Returns the subtree of the total DAG that contains only the function
    rooted at the given node.
    :param dag: The DAG which has had dag_pattern_recover() run on it.
    :param node: The node to find the function subtree of.
    :return: A NetworkX subgraph of the given DAG.
    """
    nodes = []
    _get_subtree(dag, node, nodes)
    return dag.subgraph(nodes)


def get_function(dag, node):
    """
    Return a SymPy representation of the function rooted at the given node.
    :param dag: DAG which has had dag_pattern_recover() run on it.
    :param node: The node to return the function of.
    :return: A SymPy expression of the binary function.
    """
    f = dag.node[node]['function']
    if f is None:
        return s.Symbol('[' + str(node) + ']')
    else:
        succ = dag.successors(node)
        return DS_TO_SYM[f](get_function(dag, succ[0]), get_function(dag, succ[1]))


def count_disconnected(dag):
    """
    Count the number of disconnected trees in a NetworkX DAG.
    :param dag: The DAG to count disconnected trees of.
    :return: The number of disconnected trees.
    """
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
