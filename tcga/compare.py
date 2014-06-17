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

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

################################################################################
# Mutual information / Entropy calculation


def _term(pxy, px, py):
    """
    Computes a 'term' in the binary_mutual_information() function.

    :param pxy: P(X,Y)
    :param px: P(X)
    :param py: P(Y)
    :return: P(X,Y) * log(P(X,Y)/(P(X)P(Y))), or 0 if the log is undefined.
    """
    if px == 0 or py == 0 or pxy == 0:
        return 0
    else:
        return pxy * np.log2(pxy / (px * py))


def binary_mutual_information(ds1, ds2):
    """
    Computes the mutual information (in bits) of two binary datasets.

    Uses the formula I(X;Y) = Sum[y,x] (p(x,y)log(p(x,y)/(p(x)p(y)))).  This
    isn't really the most intuitive formula for mutual information, in my mind.
    However, it seems simplest to implement.

    :param ds1: The first dataset to compute mutual information on.
    :param ds2: The second dataset to compute mutual information on.
    :return: The mutual information of ds1 and ds2.
    """
    size = len(ds1)

    # p(x=1, y=1), p(x=1, y=0), p(x=0, y=1)
    px1y1 = len(ds1[(ds1 == 1) & (ds2 == 1)]) / size
    px1y0 = len(ds1[(ds1 == 1) & (ds2 == 0)]) / size
    px0y1 = len(ds1[(ds1 == 0) & (ds2 == 1)]) / size
    px0y0 = 1 - px1y1 - px1y0 - px0y1

    # p(x=1), p(y=1)
    px1 = px1y1 + px1y0
    px0 = 1 - px1
    py1 = px0y1 + px1y1
    py0 = 1 - py1

    mutual_information = _term(px1y1, px1, py1) \
                         + _term(px1y0, px1, py0) \
                         + _term(px0y1, px0, py1) \
                         + _term(px0y0, px0, py0)
    return mutual_information


def entropy(ds, domain=(0, 1)):
    """
    Computes the entropy (in bits) of a dataset.

    :param ds: The dataset to compute the entropy of.
    :param domain: The domain of the dataset.
    :return: The entropy of the dataset, in bits.
    """
    entropy = 0
    total = len(ds)
    for value in domain:
        probability = len(ds[ds == value]) / total
        if probability != 0:  # avoid log(0)
            entropy -= probability * np.log2(probability)
    return entropy


def conditional_entropy(ds, cs, ds_domain=(0, 1), cs_domain=(0, 1)):
    """
    Computes the conditional entropy of ds given cs, AKA H(ds|cs), in bits.

    :param ds: The non-conditioned dataset (eg. X in H(X|Y)).
    :param cs: The conditioned dataset. (eg. Y in H(X|Y)).
    :param ds_domain: The domain of the non-conditioned dataset.
    :param cs_domain: The domain of the conditioned dataset.
    :return: The conditional entropy of ds given cs, AKA H(ds|cs), in bits.
    """
    entropy = 0
    total = len(ds)
    for d in ds_domain:
        for c in cs_domain:
            pBoth = len(ds[(ds == d) & (cs == c)]) / total
            pCondition = len(cs[cs == c]) / total
            if pCondition != 0:  # if pCondition == 0, pBoth == 0 too
                entropy += pBoth * np.log2(pCondition / pBoth)
    return entropy


def compare_mi_methods(ds1, ds2):
    """Computes binary mutual information multiple ways and compares.

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
        binary_mutual_information(ds1, ds2),

        entropy(ds1) - conditional_entropy(ds1, ds2),

        entropy(ds2) - conditional_entropy(ds2, ds1),

        entropy(ds1) + entropy(ds2) - entropy(combined, domain=[0, 1, 2, 3]),

        entropy(combined, domain=[0, 1, 2, 3]) - conditional_entropy(ds1, ds2)
        - conditional_entropy(ds2, ds1),
    ]
    for r1 in results:
        for r2 in results:
            if not np.isclose([r1], [r2]):
                return False
    return True


################################################################################
# Boolean functions.  All are designed to take and receive bools, not integers
# (though it wouldn't take much to change that).

# ~ 99 microseconds
def dsAnd(x, y):
    """Dataset AND.  For use with Pandas Series."""
    return x.multiply(y).astype(bool)


# ~ 192 microseconds (!)
def dsOr(x, y):
    """Dataset OR.  For use with Pandas Series."""
    return ~(~x).multiply(~y)


# ~ 59 microseconds
def dsXor(x, y):
    """Dataset XOR.  For use with Pandas Series."""
    return x != y


# ~ 99 + 46 = 145 microseconds
def dsAndNot(x, y):
    """Dataset x AND NOT y.  For use with Pandas Series."""
    return dsAnd(x, ~y)


# ~ 99 + 46 = 145 microseconds
def dsNotAnd(x, y):
    """Dataset NOT x AND y.  For use with Pandas Series."""
    return dsAnd(~x, y)


COMBINATIONS = [dsAnd, dsOr, dsXor, dsAndNot, dsNotAnd]


def best_combination(d1, d2, p):
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
    best_mutual_info = 0
    best_func = None
    best_dataset = None
    second_best_mutual_info = 0

    for func in COMBINATIONS:
        dataset = func(d1, d2)
        mutual_info = binary_mutual_information(dataset, p)
        if mutual_info >= best_mutual_info:
            second_best_mutual_info = best_mutual_info
            best_mutual_info = mutual_info
            best_func = func
            best_dataset = dataset
    return best_func, best_dataset, best_mutual_info, second_best_mutual_info


################################################################################
# Benchmarks for pattern detection

def _binary_distribution(size, dist):
    """
    Generate a boolean dataset of a particular size specified probability.

    :param size: The number of items in the dataset.
    :param dist: The probability of a 1/True.
    :return: A Series of bools randomly generated with the given parameters.
    """
    ds = Series(np.random.ranf(size))
    ds[ds < dist] = 1
    ds[ds != 1] = 0
    return ds.astype(bool)


def reclaim_pattern(size, function, proportion, dist=0.5):
    """
    Tests the best_combination function.

    Creates three random binary datasets (two data, one phenotype) with given
    size.  Adds the pattern from the given function for the given proportion of
    samples.  Then, checks whether best_combination() picks the correct
    function.  Returns True if it did, False otherwise.

    :param size: The size of the distribution to randomly create.
    :param function: The function pattern to implant into the random data.
    :param proportion: The proportion of the data to implant the pattern into.
    :param dist: The probability of a 1 for the random dataset generation.
    """
    d1 = _binary_distribution(size, dist)
    d2 = _binary_distribution(size, dist)
    p = _binary_distribution(size, dist)
    pattern = function(d1, d2)
    for i in range(int(proportion * size)):
        p[i] = pattern[i]
    res_func, *etc = best_combination(d1, d2, p)
    return res_func == function


def pattern_detection_rate(size, function, trials, start=0.3, step=0.01,
                           dist=0.5):
    """
    Finds the lowest proportion at which best_combination works.

    Uses reclaim_pattern() on the given function and size.  Starts at 30%
    pattern (by default) and moves down by 1% (by default).  Terminates when the
    best combination is not found by every trial (number of trials determined by
    the variable trials), and returns the failed proportion.

    :param size: The size of the dataset to test.
    :param function: The function (from tcga.compare.COMBINATIONS) to test.
    :param trials: The number of trials to perform at each step.
    :param start: The pattern implantation proportion to start at.
    :param step: The proportion to decrement at each step.
    :param dist: The random distribution.
    :return: The proportion at which one step failed to identify the pattern.
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
                                      step=step)
        if out:
            print('%s  %f' % (function.__name__, rate))
        rates.append(rate)
    return rates


def collect_detection_data(size_start=500, size_end=2000, size_step=250,
                           runs_per_step=25, out=True):
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
    for size in range(size_start, size_end + 1, size_step):
        if out:
            print('=> Dataset size %d.' % size)
        for x in range(runs_per_step):
            data.loc[len(data)] = [size] + all_detection_rates(size, trials=1)
    return data


def detection_by_distribution(dist_start=0.05, dist_end=1.00, dist_step=0.05,
                              runs_per_step=25, size=1000, out=True):
    """
    Collect data on the detection rate by the distribution of the dataset.

    :param dist_start: Starting detection rate.
    :param dist_end: Ending detection rate.
    :param dist_step: Increment to detection rate in sweep.
    :param runs_per_step: Number of times to run all_detection_rates() on
    each dist step.
    :param size: The size to test at.
    :param out: Whether to output progress.
    :return: A DataFrame with data for each dist x function combination.
    """
    data = DataFrame(columns=['P(1)'] + [f.__name__ for f in COMBINATIONS],
                     dtype=float)
    dist = dist_start
    while dist < dist_end:
        if out:
            print('=> Distribution: %f.' % dist)
        for x in range(runs_per_step):
            data.loc[len(data)] = [dist] + all_detection_rates(size, dist=dist)
        dist += dist_step
    return data