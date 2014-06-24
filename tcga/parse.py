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
from os.path import expanduser, expandvars

import numpy as np
import pandas as pd


MUTSLOC = '~/data/tcga-brca/mutations.csv'
PHENLOC = '~/data/tcga-brca/vital-status.csv'


def _read_csv(filename):
    """Reads a CSV file, skipping header. Returns tuples containing each row."""
    with open(filename) as f:
        r = csv.reader(f)
        next(r)
        yield from r


def mutations(filename=MUTSLOC):
    """Reads a sparse mutation file.  Returns a dense Pandas DataFrame.

    The returned DataFrame is indexed by the patients (that is, patients are the
    row), and the columns are the genes.

    """
    filename = expanduser(expandvars(filename))
    raw_muts = list(_read_csv(filename))
    genes, patients = zip(*raw_muts)  # unzip the zipped lists
    genes = set(genes)
    patients = set(patients)
    muts = pd.DataFrame(np.zeros((len(patients), len(genes))),
                        index=patients, columns=genes, dtype=bool)
    for gene, patient in raw_muts:
        muts.ix[patient, gene] = 1
    return muts


def phenotypes(filename=PHENLOC, dtype=bool):
    """Reads a mutation CSV.  Returns a pd.Series indexed by patient."""
    filename = expanduser(expandvars(filename))
    val = pd.read_csv(filename, index_col=0, squeeze=True)
    if dtype:
        return val.astype(dtype)
    else:
        return val


def add_empties(muts, phen):
    """
    Add the patients with no mutations into the mutations dataset.

    Unfortunately, this copies the data in memory (I'm not sure why).  Returns a
    2-tuple:

    [0]: modified mutations dataframe
    [1]: modified phenotypes dataframe

    """
    muts = muts.reindex(index=muts.index.union(phen.index), fill_value=False)
    phen = phen.reindex(index=muts.index, fill_value=False)
    return muts, phen


def data(mutationsloc=MUTSLOC, phenotypesloc=PHENLOC, phenotypesdtype=bool):
    """Read both data files and restrict patients to those contained in both."""
    muts = mutations(mutationsloc)
    phen = phenotypes(phenotypesloc, phenotypesdtype)
    return add_empties(muts, phen)


def sparse_mutations(filename=MUTSLOC, phenotype=None):
    """
    Reads the sparse mutations, and returns a list of (gene, patient) tuples.
    Restricts to only the patients included in the phenotype index,
    if phenotype is provided.
    :param filename: The filename of the mutations csv file.
    :param phenotype: The phenotype Series.
    :return: A Python list of (gene, patient) tuples.
    """
    filename = expanduser(expandvars(filename))
    muts = list(_read_csv(filename))
    if phenotype is not None:
        muts = [(gene, patient) for gene, patient in muts if patient in
                phenotype.index]
    return list(set(muts)) # no duplicates