"""Contains objective functions for use with tcga.pattern.dag_pattern_recover().

Objective functions for dag_pattern_recover() take two arguments: the first
is the node dataset, and the second is the phenotype dataset.  The objective
functions should return a number that will be *maximized* by the procedure.
"""

import numpy as np
from lifelines.statistics import logrank_test


def _entropy(ds, domain=(0, 1)):
    """
    Computes the _entropy (in bits) of a dataset.

    :param ds: The dataset to compute the _entropy of.
    :param domain: The domain of the dataset.
    :return: The _entropy of the dataset, in bits.
    """
    currentropy = 0
    total = len(ds)
    for value in domain:
        probability = len(ds[ds == value]) / total
        if probability != 0:  # avoid log(0)
            currentropy -= probability * np.log2(probability)
    return currentropy


def _conditional_entropy(ds, cs, ds_domain=(0, 1), cs_domain=(0, 1)):
    """
    Computes the conditional _entropy of ds given cs, AKA H(ds|cs), in bits.

    :param ds: The non-conditioned dataset (eg. X in H(X|Y)).
    :param cs: The conditioned dataset. (eg. Y in H(X|Y)).
    :param ds_domain: The domain of the non-conditioned dataset.
    :param cs_domain: The domain of the conditioned dataset.
    :return: The conditional _entropy of ds given cs, AKA H(ds|cs), in bits.
    """
    currentropy = 0
    total = len(ds)
    for d in ds_domain:
        for c in cs_domain:
            pboth = len(ds[(ds == d) & (cs == c)]) / total
            pcondition = len(cs[cs == c]) / total
            if pboth != 0:
                currentropy += pboth * np.log2(pcondition / pboth)
    return currentropy


def mutual_info(ds1, ds2, ds1domain=2, ds2domain=2):
    """
    Calculate the mutual information between two datasets.  These two datasets
    should be integer datasets in the range [0,domain).

    :param ds1: First dataset, NumPy Series
    :param ds2: Second dataset, NumPy Series
    :param ds1domain: Integer domain of first dataset: x is in [0,ds1domain)
    for all x in ds1.
    :param ds2domain: Integer domain of second dataset: x is in [0,ds2domain)
    for all x in ds1.
    :return: The mutual information between the two datasets.
    """
    combined = ds1 + ds2 * ds1domain
    return _entropy(ds1, domain=range(ds1domain)) + \
           _entropy(ds2, domain=range(ds2domain)) - \
           _entropy(combined, domain=range(ds1domain * ds2domain))


def log_rank(dataset, phenotype):
    pop1 = phenotype[dataset]
    pop2 = phenotype[~dataset]

    if len(pop1) == 0 or len(pop2) == 0:
        return 0  # This is a failed test!

    pop1_lifetime = pop1['lifetime']
    pop1_censored = pop1['censored']
    pop1_event_observed = ~pop1_censored

    pop2_lifetime = pop2['lifetime']
    pop2_censored = pop2['censored']
    pop2_event_observed = ~pop2_censored

    summary, p, res = logrank_test(pop1_lifetime, pop2_lifetime,
                                   event_observed_A=pop1_event_observed,
                                   event_observed_B=pop2_event_observed,
                                   suppress_print=True)
    return - np.log(p)


# ~ 99 microseconds
def ds_and(x, y):
    """Dataset AND.  For use with Pandas Series."""
    return x.multiply(y).astype(bool)


# ~ 192 microseconds (!)
def ds_or(x, y):
    """Dataset OR.  For use with Pandas Series."""
    return ~(~x).multiply(~y)


# ~ 59 microseconds
def ds_xor(x, y):
    """Dataset XOR.  For use with Pandas Series."""
    return x != y


# ~ 99 + 46 = 145 microseconds
def ds_and_not(x, y):
    """Dataset x AND NOT y.  For use with Pandas Series."""
    return ds_and(x, ~y)


# ~ 99 + 46 = 145 microseconds
def ds_not_and(x, y):
    """Dataset NOT x AND y.  For use with Pandas Series."""
    return ds_and(~x, y)


def ds_x(x, y):
    return x


def ds_y(x, y):
    return y


COMBINATIONS = [ds_and, ds_or, ds_xor, ds_and_not, ds_not_and, ds_x, ds_y]


def best_combination(d1, d2, p, comparison=mutual_info):
    """
    Find the best binary function of d1 and d2 to maximize comparison.

    :param d1: First dataset.
    :param d2: Second dataset.
    :param p: Phenotype dataset to correlate to a boolean function of d1 and d2.
    :param comparison: The objective function to use.
    :return: A four-tuple:
    [0] the best boolean function
    [1] the resulting dataset
    [2] the mutual information of this function
    [3] the mutual information of the second function

    """
    best_value = 0
    best_func = None
    best_dataset = None
    second_value = 0

    for func in COMBINATIONS:
        dataset = func(d1, d2)
        value = comparison(dataset, p)
        if value >= best_value:
            second_value = best_value
            best_value = value
            best_func = func
            best_dataset = dataset
    return best_func, best_dataset, best_value, second_value
