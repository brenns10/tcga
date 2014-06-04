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

def test_binary_mutual_information(dataSet1, dataSet2):
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
