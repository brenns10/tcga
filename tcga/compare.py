#-------------------------------------------------------------------------------
# 
# File:         compare.py
#
# Author:       Stephen Brennan
#
# Date Created: Wednesday,  4 June 2014
#
# Description:  Contains functions to compare binary data sets.
#
#-------------------------------------------------------------------------------

import random
import numpy as np
from pandas import Series, DataFrame
from lifelines.statistics import logrank_test

from .util import progress

################################################################################
# Mutual information / Entropy calculation


def entropy(ds, domain=(0, 1)):
    """
    Computes the entropy (in bits) of a dataset.

    :param ds: The dataset to compute the entropy of.
    :param domain: The domain of the dataset.
    :return: The entropy of the dataset, in bits.
    """
    currentropy = 0
    total = len(ds)
    for value in domain:
        probability = len(ds[ds == value]) / total
        if probability != 0:  # avoid log(0)
            currentropy -= probability * np.log2(probability)
    return currentropy


def conditional_entropy(ds, cs, ds_domain=(0, 1), cs_domain=(0, 1)):
    """
    Computes the conditional entropy of ds given cs, AKA H(ds|cs), in bits.

    :param ds: The non-conditioned dataset (eg. X in H(X|Y)).
    :param cs: The conditioned dataset. (eg. Y in H(X|Y)).
    :param ds_domain: The domain of the non-conditioned dataset.
    :param cs_domain: The domain of the conditioned dataset.
    :return: The conditional entropy of ds given cs, AKA H(ds|cs), in bits.
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
    return entropy(ds1, domain=range(ds1domain)) + \
           entropy(ds2, domain=range(ds2domain)) - \
           entropy(combined, domain=range(ds1domain * ds2domain))


def compare_mi_methods(ds1, ds2):
    """
    Computes binary mutual information multiple ways and compares.

    Uses the direct formula (binary_mutual_information()), as well as the
    following formulas which use entropy:
    * H(X) - H(X|Y)
    * H(Y) - H(Y|X)
    * H(X) + H(Y) - H(X,Y)
    * H(X,Y) - H(X|Y) - H(Y|X)
    Returns False if the mutual information functions do not return the same
    values (within floating point error).

    :param ds1: The first dataset to compute mutual information with.
    :param ds2: The second dataset to compute mutual information with.
    :return: False if the comparison fails, True otherwise.
    """
    combined = ds1 + ds2 * 2
    results = [
        entropy(ds1) - conditional_entropy(ds1, ds2),

        entropy(ds2) - conditional_entropy(ds2, ds1),

        entropy(ds1) + entropy(ds2) - entropy(combined, domain=[0, 1, 2, 3]),

        mutual_info(ds1, ds2)
    ]
    for r1 in results:
        for r2 in results:
            if not np.isclose([r1], [r2]):
                return False
    return True

################################################################################
# Log Rank Comparison

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


################################################################################
# Boolean functions.  All are designed to take and receive bools, not integers
# (though it wouldn't take much to change that).

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
    Find the best boolean combination of d1 and d2 to relate to p.

    Uses mutual information to find the best boolean function between d1 and d2
    to relate to p.

    :param d1: First dataset.
    :param d2: Second dataset.
    :param p: Phenotype dataset to correlate to a boolean function of d1 and d2.
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


################################################################################
# Benchmarks for pattern detection

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


def implant_pattern(d1, d2, phenotype, function, pattern_density):
    """
    Randomly implant a pattern f(d1, d2) into a dataset at a given pattern_density.
    :param d1: First dataset
    :param d2: Second dataset
    :param phenotype: The dataset to implant into
    :param function: The pattern function
    :param pattern_density: The pattern_density to implant at
    :return:
    """
    pattern = function(d1, d2)
    total = len(phenotype)
    amount = int(pattern_density * total)
    for i in random.sample(range(total), amount):
        phenotype[i] = pattern[i]


def reclaim_pattern(size, function, pattern_density, dist=0.5):
    """
    Tests the best_combination function.

    Creates three random binary datasets (two data, one phenotype) with given
    size.  Adds the pattern from the given function for the given proportion of
    samples.  Then, checks whether best_combination() picks the correct
    function.  Returns True if it did, False otherwise.

    :param size: The size of the distribution to randomly create.
    :param function: The function pattern to implant into the random data.
    :param pattern_density: The proportion of the data to implant the pattern into.
    :param dist: The probability of a 1 for the random dataset generation.
    """
    d1 = binary_distribution(size, dist)
    d2 = binary_distribution(size, dist)
    p = binary_distribution(size, dist)
    implant_pattern(d1, d2, p, function, pattern_density)
    res_func, *etc = best_combination(d1, d2, p)
    return res_func == function


def pattern_detection_rate(size, function, trials, start=0.3, step=0.01,
                           dist=0.5):
    """
    Finds the lowest pattern_density at which best_combination works.

    Uses reclaim_pattern() on the given function and size.  Starts at 30%
    pattern (by default) and moves down by 1% (by default).  Terminates when the
    best combination is not found by every trial (number of trials determined by
    the variable trials), and returns the failed pattern_density.

    :param size: The size of the dataset to test.
    :param function: The function (from tcga.compare.COMBINATIONS) to test.
    :param trials: The number of trials to perform at each step.
    :param start: The pattern implantation proportion to start at.
    :param step: The pattern_density to decrement at each step.
    :param dist: The random distribution.
    :return: The pattern_density at which one step failed to identify the pattern.
    """
    curr = start
    detected = True
    while detected:
        for x in range(trials):
            if not reclaim_pattern(size, function, curr, dist):
                detected = False
        if detected:
            curr -= step
    return curr


def all_detection_rates(size, trials=5, start=0.3, step=0.01, dist=0.5,
                        out=False):
    """
    Tests the detection rate for all functions.

    For each function, determines the detection rate at a specific size.
    Basically just runs pattern_detection_rates() on all functions in
    COMBINATIONS.

    :param size: The size to use for the random datasets.
    :param trials: The number of trials to use on pattern_detection_rate().
    :param start: The start for the pattern_detection_rate() sweep.
    :param step: The step for the pattern_detection_rate() sweep.
    :param dist: The random distribution.
    :param out: Whether or not to print each function along with its rate.
    :return: A list of the rates, in the same order as
    tcga.compare.COMBINATIONS.
    """
    rates = []
    for function in COMBINATIONS:
        rate = pattern_detection_rate(size, function, trials, start=start,
                                      step=step, dist=dist)
        if out:
            print('%s  %f' % (function.__name__, rate))
        rates.append(rate)
    return rates


def collect_detection_data(size_start=500, size_end=2000, size_step=250,
                           runs_per_step=25):
    """
    Collect data on the detection rate by the size of the dataset.

    :param size_start: Beginning of the size range.
    :param size_end: End of the size range.
    :param size_step: Size range step (inclusive).
    :param runs_per_step: Number of times to run all_detection_rates() on
    each size step.
    :return: A DataFrame with data for each size x function combination.
    """
    data = DataFrame(columns=['size'] +
                             [f.__name__ for f in COMBINATIONS],
                     dtype=float)
    for size in progress(range(size_start, size_end + 1, size_step)):
        for x in range(runs_per_step):
            data.loc[len(data)] = [size] + all_detection_rates(size, trials=1)
    return data


def detection_by_distribution(dist_start=0.05, dist_end=1.00, dist_step=0.05,
                              runs_per_step=25, size=1000):
    """
    Collect data on the detection rate by the distribution of the dataset.

    :param dist_start: Starting detection rate.
    :param dist_end: Ending detection rate.
    :param dist_step: Increment to detection rate in sweep.
    :param runs_per_step: Number of times to run all_detection_rates() on
    each dist step.
    :param size: The size to test at.
    :return: A DataFrame with data for each dist x function combination.
    """
    data = DataFrame(columns=['P(1)'] + [f.__name__ for f in COMBINATIONS],
                     dtype=float)
    for dist in progress(range(dist_start, dist_end, dist_step)):
        for x in range(runs_per_step):
            data.loc[len(data)] = [dist] + all_detection_rates(size, dist=dist)
    return data
