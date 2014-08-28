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
import networkx as nx

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


def dag_pattern_recover(muts, phen, dag):
    """
    Run the pattern detection algorithm on the given data.

    This function starts at the leaf nodes of the DAG, which are genes.  It
    moves up through the tree and computes the best binary function to relate
    the child nodes, using the mutual information between the phenotype and
    the function value between the children.  It stores the resulting
    function, dataset, and mutual information as attributes in the nodes.

    Note that not all genes in the DAG have mutation information associated
    with them.  In this case, the algorithm ignores these children.

    :param muts: The mutation dataset (as pandas DataFrame).
    :param phen: The phenotype dataset (as pandas DataFrame).
    :param dag: The DAG (as NetworkX digraph).
    :return: None
    """
    leaves = (n for n, d in dag.out_degree_iter() if d == 0)
    nodes = collections.deque()

    nx.set_node_attributes(dag, 'visited', False)
    nx.set_node_attributes(dag, 'dataset', None)
    nx.set_node_attributes(dag, 'function', None)
    nx.set_node_attributes(dag, 'mutual_info', None)

    # Look at all leaf nodes, and check whether we have mutation data for
    # them.  If they do, save a reference to that dataset in an attribute.
    # Else, store None.
    for leaf in leaves:
        # Set up the dataset parameter.
        params = dag.node[leaf]
        if leaf not in muts.columns:
            params['dataset'] = None
        else:
            params['dataset'] = leaf
        params['visited'] = True

        # Add its parent to the iteration queue, if all its children are leaves
        for pred in dag.predecessors(leaf):
            if all(dag.node[succ]['visited'] for succ in dag.successors(pred)):
                nodes.appendleft(pred)

    # Let the breadth first search begin, but in reverse!
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

        # The children may not have been included in the mutation datasets.
        # Check for that here.
        if x_key is None:
            if y_key is None:
                # Neither child has a dataset.
                params['dataset'] = None
            else:
                # Y has a dataset, but not X.
                params['dataset'] = y_key
                params['function'] = compare.ds_y
                params['mutual_info'] = dag.node[children[1]]['mutual_info']
        else:
            if y_key is None:
                # X has a dataset, but not Y.
                params['dataset'] = x_key
                params['function'] = compare.ds_x
                params['mutual_info'] = dag.node[children[0]]['mutual_info']
            else:
                # Both have datasets.  This is the normal case.
                function, dataset, mi, *etc = compare.best_combination(
                    muts[x_key], muts[y_key], phen)
                params['function'] = function
                params['dataset'] = curr
                muts[curr] = dataset
                params['mutual_info'] = mi

        # Add (each) parent if all the parent's children have been visited.
        for pred in dag.predecessors(curr):
            if all(dag.node[succ]['visited'] for succ in dag.successors(pred)):
                nodes.appendleft(pred)


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
        mi = dag.node[root]['mutual_info']
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

    return sparse