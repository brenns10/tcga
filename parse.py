#-------------------------------------------------------------------------------
# 
# File:         parse.py
#
# Author:       Stephen Brennan, Matthew Ruffalo
#
# Date Created: Tuesday,  3 June 2014
#
# Description:  Contains functions for parsing TCGA data.
#
#-------------------------------------------------------------------------------

import csv
import numpy as np
import pandas as pd

def _read_csv(filename):
    """Reads a CSV file, skipping header. Returns tuples containing each row."""
    with open(filename) as f:
        r = csv.reader(f)
        next(r)
        yield from r

def mutations(filename):
    """Reads a sparse mutation file.  Returns a dense Pandas DataFrame.

    The returned DataFrame is indexed by the patients (that is, patients are the
    row), and the columns are the genes.

    """
    raw_muts = list(_read_csv(filename))
    genes, patients = zip(*raw_muts) # unzip the zipped lists
    genes = set(genes)
    patients = set(patients)
    mutations = pd.DataFrame(np.zeros((len(patients), len(genes))), 
                             index=patients, columns=genes, dtype=bool)
    for gene, patient in raw_muts:
        mutations.ix[patient, gene] = 1
    return mutations

def phenotypes(filename, dtype=None):
    """Reads a mutation CSV.  Returns a pd.Series indexed by patient."""
    val = pd.read_csv(filename, index_col=0, squeeze=True)
    if dtype:
        return val.astype(dtype)
    else:
        return val

def restrict(mutations, phenotypes):
    """Restrict the patients to only those contained in both datasets.

    Unfortunately, this copies the data in memory (I'm not sure why).  Returns a
    2-tuple:

    [0]: modified mutations dataframe
    [1]: modified phenotypes dataframe

    """
    mutations = mutations.reindex(index=mutations.index.intersection(phenotypes.index))
    phenotypes = phenotypes.reindex(index=mutations.index)
    return mutations, phenotypes

