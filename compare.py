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
    if px == 0 or py == 0 or pxy == 0:
        return 0
    else:
        return pxy * np.log2(pxy / (px * py))

def binary_mutual_information(dataSet1, dataSet2):
    """Computes the mutual information (in bits) of two binary datasets.

    Uses the formula I(X;Y) = Sum[y,x] (p(x,y)log(p(x,y)/(p(x)p(y)))).  This
    isn't really the most intuitive formula for mutual information, in my mind.
    However, it seems simplest to implement.

    """
    totalData = len(dataSet1)

    # p(x=1, y=1), p(x=1, y=0), p(x=0, y=1)
    px1y1 = len(dataSet1[(dataSet1 == 1) & (dataSet2 == 1)]) / totalData
    px1y0 = len(dataSet1[(dataSet1 == 1) & (dataSet2 == 0)]) / totalData
    px0y1 = len(dataSet1[(dataSet1 == 0) & (dataSet2 == 1)]) / totalData
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

def entropy(dataSet, domain = [0, 1]):
    """Computes the entropy (in bits) of a dataset."""
    entropy = 0
    total = len(dataSet)
    for value in domain:
        probability = len(dataSet[dataSet == value]) / total
        if probability != 0: # avoid log(0)
            entropy -= probability * np.log2(probability)
    return entropy

def conditional_entropy(dataSet, conditionSet, dsDomain=[0,1], csDomain=[0,1]):
    """Computes the conditional entropy of dataSet given conditionSet."""
    entropy = 0
    total = len(dataSet)
    for d in dsDomain:
        for c in csDomain:
            pBoth = len(dataSet[(dataSet == d) & (conditionSet == c)]) / total
            pCondition = len(conditionSet[conditionSet == c]) / total
            if pCondition != 0: # if pCondition == 0, pBoth == 0 too
                entropy += pBoth * np.log2(pCondition / pBoth)
    return entropy

def compare_mi_methods(dataSet1, dataSet2):
    """Computes binary mutual information multiple ways and compares.

    Uses the direct formula (binary_mutual_information()), as well as the
    following formulas which use entropy:
    * H(X) - H(X|Y)
    * H(Y) - H(Y|X)
    * H(X) + H(Y) - H(X,Y)
    * H(X,Y) - H(X|Y) - H(Y|X)

    """
    combined = dataSet1 + dataSet2 * 2
    results = [
        binary_mutual_information(dataSet1, dataSet2),
        
        entropy(dataSet1) - conditional_entropy(dataSet1, dataSet2),
        
        entropy(dataSet2) - conditional_entropy(dataSet2, dataSet1),

        entropy(dataSet1) + entropy(dataSet2) - entropy(combined, 
                                                        domain=[0,1,2,3]),

        entropy(combined, domain=[0,1,2,3])           \
            - conditional_entropy(dataSet1, dataSet2) \
            - conditional_entropy(dataSet2, dataSet1)
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
    """Find the best boolean combination of d1 and d2 to relate to p.

    Uses mutual information to find the best boolean function between d1 and d2
    to relate to p.  Returns a four-tuple:

    [0] the best boolean function
    [1] the resulting dataset
    [2] the mutual information of this function
    [3] the mustual information of the second function

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

def reclaim_pattern(size, function, proportion):
    """Tests the best_combination function.

    Creates three random binary datasets (two data, one phenotype) with given
    size.  Adds the pattern from the given function for the given proportion of
    samples.  Then, checks whether best_combination() picks the correct
    function.  Returns True if it did, False otherwise.

    """
    d1 = Series(np.random.randint(0, 2, size)).astype(bool)
    d2 = Series(np.random.randint(0, 2, size)).astype(bool)
    p = Series(np.random.randint(0, 2, size)).astype(bool)
    pattern = function(d1, d2)
    for i in range(int(proportion * size)):
        p[i] = pattern[i]
    res_func, *etc = best_combination(d1, d2, p)
    return res_func == function

def pattern_detection_rate(size, function, trials, start=0.3, step=0.01):
    """Finds the lowest proportion at which best_combination works.

    Uses reclaim_pattern() on the given function and size.  Starts at 30%
    pattern (by default) and moves down by 1% (by default).  Terminates when the
    best combination is not found by every trials, and returns the failed
    proportion.

    """
    curr = start
    step = 0.01
    detected = True
    while detected:
        for x in range(trials):
            if not reclaim_pattern(size, function, curr):
                detected = False
        if detected:
            curr -= step
    return curr

def all_detection_rates(size, trials=5, start=0.3, step=0.01, out=False):
    """Tests the detection rate for all functions."""
    rates = []
    for function in COMBINATIONS:
        rate = test_detection_rate(size, function, trials, start=start, step=step)
        if out:
            print('%s  %f' % (function.__name__, rate))
        rates.append(rate)
    return rates
