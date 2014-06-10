#-------------------------------------------------------------------------------
# 
# File:         tree.py
#
# Author:       Stephen Brennan
#
# Date Created: Wednesday,  4 June 2014
#
# Description:  Chooses best binary functions on trees.
#
#-------------------------------------------------------------------------------

from . import compare
import random
import sympy as s

# These are functions that take expressions and combine them using whatever
# symbol they are named for.
def symAnd(x, y):
    return x & y
def symOr(x, y):
    return x | y
def symXor(x, y):
    return x ^ y
def symAndNot(x, y):
    return x & ~y
def symNotAnd(x, y):
    return ~x & y

# This dictionary maps the dataset functions to the symbolic functions, so we
# can build expressions from the tree we run.
DS_TO_SYM = {
    compare.dsAnd:symAnd,
    compare.dsOr:symOr,
    compare.dsXor:symXor,
    compare.dsAndNot:symAndNot,
    compare.dsNotAnd:symNotAnd,
}

def randTree(muts, phen, depth, verbose=True, simplify=False):
    """Create a boolean expression for a random tree of genes.

    For the given mutation and phenotype data, construct a random tree/ontology
    of genes, and find the boolean expression that best relates the mutations to
    the phenotype.  Returns the SymPy expression generated.

    muts - Mutation data in a pandas DataFrame
    phen - Phenotype data in a pandas Series
    depth - The depth of the tree.  Must be >=1
    verbose - Print info about what is happening
    simplify - Do we simplify the expression before returning (can be lengthy)
    """
    # Create a random sample from the entire list of possible genes
    if verbose: print("Selecting gene sample ...")  
    choices = iter(random.sample(range(len(muts.columns)), depth + 1))

    # Get two genes from the sample (as string names).
    firstStr = muts.columns[next(choices)]
    secondStr = muts.columns[next(choices)]

    # Creae sympy symbols from the gene names
    firstSym = s.Symbol('['+firstStr+']')
    secondSym = s.Symbol('['+secondStr+']')

    # Get the actual Series for each gene
    first = muts[firstStr]
    second = muts[secondStr]
    
    # Choose our first combination
    print("Choosing between %s and %s." % (firstStr, secondStr))
    func, prev, mi1, mi2 = compare.best_combination(first, second, phen)
    print("Selecting %s, MI=%f, by margin of %f." % (func.__name__, mi1, mi1-mi2))

    # Construct an expression using the function returned
    expr = DS_TO_SYM[func](firstSym, secondSym)

    # Now, begin the loop for the rest of the random tree.  In each iteration,
    # get the next gene from the random sample, find the best combination with
    # the existing dataset function, and add it to the result expression.
    for curr in choices:
        currStr = muts.columns[curr]
        print("=> Adding %s." % (currStr))
        currDataSet = muts[currStr]
        func, prev, mi1, mi2 = compare.best_combination(prev, currDataSet, phen)
        print("   Selecting %s, MI=%f, by margin of %f." % (func.__name__, mi1, mi1-mi2))
        expr = DS_TO_SYM[func](expr, s.Symbol('['+currStr+']'))

    if simplify:
        expr = s.simplify_logic(expr, 'dnf')
    if verbose:
        print('Expression constructed:')
        s.pprint(expr)
    return expr
