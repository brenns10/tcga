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


def _initialize_node_attributes(dag):
    """
    Set the node attributes used by dag_pattern_recover to default values.
    :param dag: The DAG to set attributes of.
    :return: None
    """
    # Whether each node has been visited by the search yet.
    nx.set_node_attributes(dag, 'visited', False)
    # The title of the row in the mutations dataset corresponding to this node.
    nx.set_node_attributes(dag, 'dataset', None)
    # The function chosen at this (internal) node.
    nx.set_node_attributes(dag, 'function', None)
    # The mutual information of this node's dataset with the phenotype.
    nx.set_node_attributes(dag, 'mutual_info', None)
    # The number of genes in the subtree rooted at this node.
    nx.set_node_attributes(dag, 'genes', None)


def _update_max_mutual_info(k, mi, by_gene):
    """
    Update the max mutual info accounting in by_gene.
    :param k: The k value for the node.
    :param mi: The mutual info of the node.
    :param by_gene: The dictionary the data is stored at.
    :return: None.
    """
    # Update the max mutual info.
    if mi is not None:
        by_gene[k] = max(by_gene.get(k, 0), mi)


def _add_parent_nodes(node, dag, node_queue):
    """
    Add a node's parents to the queue.  Only does so if the parent has all
    its successors visited.
    :param node: Node whose parents to add to queue.
    :param dag: DAG the node is from.
    :param node_queue: Queue to add parents to.
    :return: None
    """
    for pred in dag.predecessors(node):  # could have multiple parents
        if all(dag.node[succ]['visited'] for succ in dag.successors(pred)):
            node_queue.appendleft(pred)


def _initialize_leaves(muts, phen, dag, by_gene):
    """
    Create a queue initialized with the leaves of the DAG.

    This function is a subroutine of dag_pattern_recover.  It initializes the
    leaf nodes with the correct dataset, mutual_info attributes.  Also,
    it creates a queue containing each leaf of the DAG, so that a reverse
    breadth first search can be done.
    :param muts: Mutation dataset (needed for dataset attribute).
    :param phen: Phenotype dataset (needed for mutual_info attribute).
    :param dag: DAG (needed in order to set attributes).
    :param by_gene: Dictionary that keeps track of the best mutual info by
    gene count.
    :return: A queue containing the leaves of the DAG.
    """
    nodes = collections.deque()
    leaves = (n for n, d in dag.out_degree_iter() if d == 0)

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
            params['mutual_info'] = compare.mutual_info(muts[leaf], phen)
        params['visited'] = True

        # Set the number of genes in the subtree.
        params['genes'] = 1

        # Record the maximum mutual information for a subtree of size 1.
        _update_max_mutual_info(1, params['mutual_info'], by_gene)

        _add_parent_nodes(leaf, dag, nodes)

    return nodes


def _compare_and_set_attributes(curr, dag, muts, phen):
    """
    Compare the datasets X and Y, and set the attributes of their parent.

    This function is basically the body of the loop of dag_pattern_recover.
    It gets the datasets for X and Y.  If both exist, it compares them,
    and selects their best combination.  If only one exists, then it selects
    that one.  If none exist, it does nothing.
    :param curr: The current node.
    :param dag: The DAG the node is in.
    :param muts: The mutations dataset.
    :param phen: The phenotype dataset.
    :return: The mutual information of the chosen function.
    """
    params = dag.node[curr]

    # Get the children of this node
    children = dag.successors(curr)
    if len(children) != 2:
        raise Exception('Invalid degree of node ' + str(curr))

    x_params = dag.node[children[0]]
    y_params = dag.node[children[1]]
    x_key = x_params['dataset']
    y_key = y_params['dataset']
    value = None

    if x_key is None:
        if y_key is None:
            # Neither child has a dataset.
            params['dataset'] = None
        else:
            # Y has a dataset, but not X.
            params['genes'] = y_params['genes']
            params['dataset'] = y_key
            params['function'] = compare.ds_y
            params['mutual_info'] = y_params['mutual_info']
    else:
        if y_key is None:
            # X has a dataset, but not Y.
            params['genes'] = x_params['genes']
            params['dataset'] = x_key
            params['function'] = compare.ds_x
            params['mutual_info'] = x_params['mutual_info']
        else:
            # Both have datasets.  This is the normal case.
            params['genes'] = x_params['genes'] + y_params['genes']
            function, dataset, value, *etc = compare.best_combination(
                muts[x_key], muts[y_key], phen)
            params['function'] = function
            params['dataset'] = curr
            muts[curr] = dataset
            params['mutual_info'] = value
    return value


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
    :return: A dictionary containing the best mutual information for every
    value of k (k=#of genes in the node's function).
    """
    by_gene = {}
    # Set the node attributes used by this function to default values.
    _initialize_node_attributes(dag)
    # Initialize the leaves and add them to a queue.
    nodes = _initialize_leaves(muts, phen, dag, by_gene)

    # Do a reverse breadth-first-search, starting at the leaves and working up.
    while nodes:
        # Get the current node and mark it as visited.
        curr = nodes.pop()
        params = dag.node[curr]
        params['visited'] = True

        # Perform the comparison (returns mutual information).
        value = _compare_and_set_attributes(curr, dag, muts, phen)
        # Update the accounting of the best mutual_info.
        _update_max_mutual_info(params['genes'], value, by_gene)
        # Add parents to the queue if they're ready.
        _add_parent_nodes(curr, dag, nodes)
    return by_gene


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
