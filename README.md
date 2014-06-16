# TCGA Data Analysis Package

## Summary

This package parses and performs data analysis on binary gene mutation data.  It
uses mutual information calculations to determine the best binary function to
relate mutation data from two genes to a binary phenotype, such as vital status.

In order to reduce the search space, a binarized gene ontology will be used.
Binary functions will be evaluated up the tree.

## Dependencies

- Python 3 (currently using latest, 3.4)
- Pandas
- NumPy
- SymPy

## Unit Tests

Unit tests are in tcga/test.py.  They can be run with the command:

```
#!bash
$ python -m unittest tcga.test
```

You need to be in the same directory as this file in order to run the unit tests.  
They will not run from inside the module.

## Important Functions

This package is first and foremost a collection of functions, not a program.
The important functions/objects are:

- `tcga.parse.data(mutationsLoc, phenotypesLoc)`: Imports data from the two
  files, and restricts the data to only patients that occur in both data frames.

- `tcga.compare.binary_mutual_information(dataSet1, dataSet2)`: Calculates the
  mutual information between two binary variables.

- `tcga.compare.entropy(dataSet, domain=[0,1])`: Calculates the entropy of a
  single (by default binary) variable.

- `tcga.compare.entropy(dataSet, conditionSet, dsDomain=[0,1], csDomain=[0,1])`:
  Calculates conditional entropy of one dataset given another.

- `tcga.compare.ds***`: Functions representing boolean combinations of datasets.

- `tcga.compare.COMBINATIONS`: A list of binary functions we check.

- `tcga.compare.best_combination(d1, d2, p)`: Determines the best binary
  function (or combination) of two variables to correlate to a phenotype `p`,
  using the mutual information function.

- `tcga.compare.reclaim_pattern(size, function, proportion)`: Creates random
  datasets of size `size`, implants a pattern (`function`) with the given
  proportion `proportion`.  Then, returns true if `best_combination()` returns
  the function implanted.

- `tcga.tree.sym***`: Functions representing boolean combinations of SymPy
  expressions.

- `tcga.tree.randTree(muts, phen, depth, verbose=True, simplify=False)`:
  Determines boolean expressions on a randomly generated, sequential tree.
