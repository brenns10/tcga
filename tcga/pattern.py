# -*- coding: utf-8 -*-
"""Contains dag_pattern_recover() and related private functions.

Author: Stephen Brennan
"""

import collections

import networkx as nx

from tcga import compare


def dag_pattern_recover(muts, phen, dag, comparison=compare.mutual_info):
    """
    Run the pattern detection algorithm on the given data.

    This function starts at the leaf nodes of the DAG, which are genes.  It
    moves up through the tree and computes the binary function between the
    two children that maximizes the objective function (`comparison`).  It
    stores the resulting function, dataset, and objective function value as
    attributes in the nodes.

    Note that not all genes in the DAG have mutation information associated
    with them.  In this case, the algorithm ignores these children.

    :param muts: The mutation dataset (as pandas DataFrame).
    :param phen: The phenotype dataset (as pandas DataFrame).
    :param dag: The DAG (as NetworkX digraph).
    :param comparison: The objective function we attempt to maximize.
    :return: A dictionary containing the best mutual information for every
    value of k (k=#of genes in the node's function).
    """
    by_gene = {}
    # Set the node attributes used by this function to default values.
    _initialize_node_attributes(dag)
    # Initialize the leaves and add them to a queue.
    nodes = _initialize_leaves(muts, phen, dag, by_gene, comparison)

    # Do a reverse breadth-first-search, starting at the leaves and working up.
    while nodes:
        # Get the current node and mark it as visited.
        curr = nodes.pop()
        params = dag.node[curr]
        params['visited'] = True

        # Perform the comparison (returns mutual information).
        value = _compare_and_set_attributes(curr, dag, muts, phen, comparison)
        # Update the accounting of the best mutual_info.
        _update_max_value(params['genes'], value, by_gene)
        # Add parents to the queue if they're ready.
        _add_parent_nodes(curr, dag, nodes)
    return by_gene


def _compare_and_set_attributes(curr, dag, muts, phen, comparison):
    """
    Determine the binary function for a node and set its attributes.

    This function is basically the body of the loop of dag_pattern_recover.
    It gets children of `curr`, X and Y.  If both exist, it compares them,
    and selects their best combination.  If only one exists, then it selects
    that one.  If none exist, it does nothing.
    :param curr: The current node.
    :param dag: The DAG the node is in.
    :param muts: The mutations dataset.
    :param phen: The phenotype dataset.
    :param comparison: The objective function for choosing the binary function.
    :return: The objective function value of the chosen binary function.
    """
    params = dag.node[curr]

    # Get the children of this node
    children = dag.successors(curr)

    assert len(children) == 2, "Tree node with #children != 2."

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
            params['value'] = y_params['value']
    else:
        if y_key is None:
            # X has a dataset, but not Y.
            params['genes'] = x_params['genes']
            params['dataset'] = x_key
            params['function'] = compare.ds_x
            params['value'] = x_params['value']
        else:
            # Both have datasets.  This is the normal case.
            params['genes'] = x_params['genes'] + y_params['genes']
            function, dataset, value, *etc = compare.best_combination(
                muts[x_key], muts[y_key], phen, comparison)
            params['function'] = function
            params['dataset'] = curr
            muts[curr] = dataset
            params['value'] = value

    return value


def _initialize_leaves(muts, phen, dag, by_gene, comparison):
    """
    Create a queue initialized with the leaves of the DAG.

    This function is a subroutine of dag_pattern_recover.  It initializes the
    leaf nodes with the correct dataset, mutual_info attributes.  Also,
    it creates a queue containing each leaf of the DAG, so that a reverse
    breadth first search can be done.
    :param muts: Mutation dataset (needed for dataset attribute).
    :param phen: Phenotype dataset (needed for mutual_info attribute).
    :param dag: DAG (needed in order to set attributes).
    :param by_gene: Dictionary that keeps track of the best objective
    function value by gene count.
    :param comparison: The objective function to use.
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
            params['value'] = comparison(muts[leaf], phen)
        params['visited'] = True

        # Set the number of genes in the subtree.
        params['genes'] = 1

        # Record the maximum mutual information for a subtree of size 1.
        _update_max_value(1, params['value'], by_gene)

        _add_parent_nodes(leaf, dag, nodes)

    return nodes


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


def _update_max_value(k, mi, by_gene):
    """
    Update the max objective function value in the by_gene dict.
    :param k: The k value for the node.
    :param mi: The mutual info of the node.
    :param by_gene: The dictionary the data is stored at.
    :return: None.
    """
    # Update the max mutual info.
    if mi is not None:
        by_gene[k] = max(by_gene.get(k, 0), mi)


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
    nx.set_node_attributes(dag, 'value', None)
    # The number of genes in the subtree rooted at this node.
    nx.set_node_attributes(dag, 'genes', None)