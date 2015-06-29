"""Simulation experiment for pattern recovery at scale."""

from util import binary_distribution
from pattern import dag_pattern_recover


def random_mutations(samples, sparsity, genes):
    pass


def create_ontology(genes):
    pass


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
