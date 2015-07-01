"""Contains useful functions and classes."""

from pandas import Series
import numpy as np
import networkx as nx


def binary_distribution(size, sparsity):
    """
    Generate a boolean dataset of a particular size specified probability.

    :param size: The number of items in the dataset.
    :param sparsity: The probability of a 1/True.
    :return: A Series of bools randomly generated with the given parameters.
    """
    ds = Series(np.random.ranf(size))
    ds[ds < sparsity] = 1
    ds[ds != 1] = 0
    return ds.astype(bool)


def dataframe_append(dataframe, rowdict):
    """
    Shortcut method for appending a row to a DataFrame.
    :param dataframe: The DataFrame to append to.
    :param rowdict: A dictionary containing each column's value.
    """
    newrow = len(dataframe)
    dataframe.loc[newrow] = 0  # init with 0's
    for k, v in rowdict.items():
        dataframe.loc[newrow, k] = v


def dag_from_heap(iterable, reverse_edges=False):
    g = nx.DiGraph()
    nodes = list(iterable)
    for i, node in enumerate(nodes):
        g.add_node(node)
        if i == 0:
            continue  # no parents of root
        parent = nodes[(i-1)//2]
        if reverse_edges:
            node, parent = parent, node
        g.add_edge(node, parent)
    return g
