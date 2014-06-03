import csv
import numpy as np
import pandas as pd

def read_mutation_file(filename):
    """
    Reads a mutation file.  Yields 2-tuples:
    
    [0] Gene name
    [1] Patient ID
    """
    with open(filename) as f:
        r = csv.reader(f)
        next(r)
        yield from r
        
def import_tcga_mutations(filename):
    raw_muts = list(read_mutation_file(filename))
    genes, patients = zip(*raw_muts)
    genes = set(genes)
    patients = set(patients)
    mutations = pd.DataFrame(np.zeros((len(patients), len(genes))), index=patients, columns=genes)
    for gene, patient in raw_muts:
        mutations.ix[patient, gene] = 1
    return mutations
