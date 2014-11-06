"""Tree and DAG related functions.

This module contains functions useful for examining the DAG produced by
tcga.pattern.dag_pattern_recover().
"""

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


def get_roots(dag):
    """
    Returns the roots of a forest of DAGs.
    :param dag: A NetworkX directed graph.
    :return: A generator that yields each root node in the DAG.
    """
    return (x for x, d in dag.in_degree_iter() if d == 0)


def get_max_root(dag):
    """
    Return the DAG root that has the highest objective function value associated
    with it (after calling dag_pattern_recover() on the DAG).
    :param dag: The NetworkX DAG, which has already had dag_pattern_recover()
    called on it.
    :return: The root with the highest objective function value.
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


def terminal_function_nodes(dag):
    """
    A generator that takes a DAG and yields 'terminal function nodes'.

    A terminal function node is a node that is at the root of a completed
    function.  A function can be completed either by:
      1. Being at the root of a tree in the DAG.
      2. Being ignored by its parent.
    :param dag: The DAG, processed by dag_pattern_recover().
    :return: Yields terminal function nodes.
    """

    for node in dag.nodes_iter():

        # We should ignore any node that has no objective function value.
        if dag.node[node]['value'] is None:
            continue

        # Node is terminal if it has no parent (is root).
        if len(dag.predecessors(node)) == 0:
            yield node

        # Node is terminal if it is being 'ignored' by its parent.
        for parent in dag.predecessors(node):

            # If the node is X, and the parent is using ds_y, then the parent
            # is 'ignoring' the node.
            if node == dag.successors(parent)[0] and \
                    dag.node[parent]['function'] == 'ds_y':

                yield node
                break  # Don't double report a node.

            # Similarly for Y.
            elif node == dag.successors(parent)[1] and \
                    dag.node[parent]['function'] == 'ds_y':

                yield node
                break  # Don't double report a node.