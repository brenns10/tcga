"""Simulation experiment for pattern recovery at scale."""

from collections import deque
import random

import pandas as pd
import networkx as nx

from tcga.util import binary_distribution
from tcga.pattern import dag_pattern_recover
from tcga.compare import COMBINATIONS


def random_mutations(samples, sparsity, genes):
    """
    Return a randomly generated matrix of mutations.

    Randomly generates a completely uniformly random matrix, using the sparsity
    as the probabilty of True.  The return value is a Pandas DataFrame
    containing genes as columns and samples as rows.

    :param samples: number of samples (patients)
    :param sparsity: probability of a mutation in each cell
    :param genes: number of genes
    :return: Gene x patient matrix of mutations
    """
    df = pd.DataFrame()
    sl = ['sample_%d' % n for n in range(samples)]
    for x in range(genes):
        s = binary_distribution(samples, sparsity)
        s.index = sl
        s.name = 'gene_%d' % x
        df['gene_%d' % x] = s
    return df


def create_ontology(genes):
    """
    Create a "gene ontology" from a list of genes.

    :param genes: Iterable containing gene names.
    :return: NetworkX digraph of the "binary ontology".
    """
    trees = deque()

    # Create a singleton tree for each graph
    for gene in genes:
        digraph = nx.DiGraph
        digraph.add_node(gene)
        digraph.node[gene]['successors'] = 1
        trees.append((gene, digraph))

    # Combine trees into a single binary tree.
    term = 0
    while len(trees) > 1:
        # Get two trees to combine
        lhs_root, lhs_tree = trees.popleft()
        rhs_root, rhs_tree = trees.popleft()

        # Come up with a new term name
        new_root = 'term_%d' % term
        term += 1

        # Copy nodes into a single tree
        lhs_tree.add_nodes_from(rhs_tree)
        lhs_tree.add_edges_from(rhs_tree)

        # Create new term, and make it the parent of the two existing trees
        lhs_tree.add_node(new_root)
        lhs_tree.add_edge(new_root, lhs_root)
        lhs_tree.add_edge(new_root, rhs_root)
        lhs_tree.nodes[new_root]['successors'] = (
            lhs_tree.nodes[lhs_root]['successors'] +
            lhs_tree.nodes[rhs_root]['successors']
        )

        # Add to the end of the queue
        trees.append((new_root, lhs_tree))
    return trees.pop()[1]


def create_function(genes, G, num_genes=4):
    """
    Create a function using the gene ontology.
    """
    terms = [n for n in G if G.node[n]['successors'] == num_genes]
    term = random.choice(terms)
    q = collections.deque([term])
    nodes = set()
    while q:
        n = q.popleft()
        nodes.add(n)
        q.extend(n.successors())
    SG = G.subgraph(nodes).copy()
    for node in SG:
        if SG.out_degree(node) != 2:
            SG.nodes[node]['function'] = random.choice(COMBINATIONS)
    return SG


def implant_pattern(genes, phenotype, function):
    pattern = 


def simulation(samples=500, sparsity=0.5, genes=1000):
    mutations = random_mutations(samples, sparsity, genes)
    phenotype = binary_distribution(samples, sparsity)
    ontology = create_ontology(mutations.columns)
    function = create_function(mutations.columns, ontology)
    implant_pattern(mutations, phenotype, function)
    dag_pattern_recover(mutations, phenotype, ontology)

    # TODO: Analyze ontology after to see if the function is the best.
