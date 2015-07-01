"""Simulation experiment for pattern recovery at scale."""

import pandas as pd

from tcga.util import binary_distribution
from tcga.pattern import dag_pattern_recover


def random_mutations(samples, sparsity, genes):
    df = pd.DataFrame()
    sl = ['sample_%d' % n for n in range(samples)]
    for x in range(genes):
        s = binary_distribution(samples, sparsity)
        s.index = sl
        s.name = 'gene_%d' % x
        df['gene_%d' % x] = s
    return df


def create_ontology(genes):
    


def create_function(genes, ontology):
    pass


def implant_pattern(genes, phenotype, function):
    pass


def simulation(samples=500, sparsity=0.5, genes=1000):
    mutations = random_mutations(samples, sparsity, genes)
    phenotype = binary_distribution(samples, sparsity)
    ontology = create_ontology(mutations)
    function = create_function(mutations, ontology)
    implant_pattern(mutations, phenotype, function)
    dag_pattern_recover(mutations, phenotype, ontology)

    # TODO: Analyze ontology after to see if the function is the best.
