"""Contains objective functions and logic functions.

Objective functions for :func:`tcga.pattern.dag_pattern_recover()` take two
arguments: the first is the node dataset, and the second is the phenotype
dataset.  The objective functions should return a number that will be
*maximized* by the procedure.

Additionally, this module contains the logic functions used for bool Series.
"""

import numpy as np
from lifelines.statistics import logrank_test

from smbio.math.information import mutual_info


def log_rank(dataset, phenotype):
    """
    Objective function based on the log rank test.

    This objective function assumes that the phenotype is a continuous
    variable that corresponds to lifetime data.  It uses the dataset to
    partition the patients into two groups, and performs log rank test on the
    groups.  The objective function is -log(P-value).

    :param Series dataset: Boolean Series
    :param phenotype: Integer/Float Series of lifetime phenotype.
    :return: -log(p-value) of log-rank test.
    :rtype: float
    """
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


def ds_x(x, _):
    return x


def ds_y(_, y):
    return y


COMBINATIONS = [ds_and, ds_or, ds_xor, ds_and_not, ds_not_and, ds_x, ds_y]


def best_combination(d1, d2, p, comparison=mutual_info):
    """
    Find the best binary function of d1 and d2 to maximize comparison.

    :param Series d1: First dataset.
    :param Series d2: Second dataset.
    :param Series p: Phenotype dataset to correlate to a boolean function of d1
    and d2.
    :param FunctionType comparison: The objective function to use.
    :rtype: tuple
    :return: A four-tuple:
    - [0] the best boolean function
    - [1] the resulting dataset
    - [2] the mutual information of this function
    - [3] the mutual information of the second function

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
