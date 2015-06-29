TCGA Data Analysis Package
==========================

Summary
-------

This package implements an algorithm for inference of Boolean functions of
somatic mutations that correlate with phenotypes.  It uses mutual information as
an objective function, and uses the Gene Ontology (as a binary tree) to
structure and limit the search space for Boolean functions.

Project Technologies and Dependencies
-------------------------------------

* Python 3
* [smbio](http://singularity.case.edu/sbrennan/smbio) (also on
  [GitHub](https://github.com/brenns10/smbio)): a library containing helpful,
  reusable bioinformatics code for Python.
* [Pandas](https://pypi.python.org/pypi/pandas): a library for loading, saving,
  manipulating, and querying large numeric datasets quickly in Python.
* [NumPy](https://pypi.python.org/pypi/numpy): a library for performing numeric
  computations in Python.
* [SymPy](https://pypi.python.org/pypi/sympy): a library for manipulation of
  symbolic representations of math formulae in Python.  Used only for outputting
  a "pretty" representation of result functions.
* [NetworkX](https://pypi.python.org/pypi/networkx): a library for manipulation
  of graphs in Python.
* [Lifelines](https://pypi.python.org/pypi/lifelines) a library for computing
  log-rank test statistics in Python.  This will likely be removed soon.

Structure
---------

### Framework

The framework of the pattern recovery algorithm is contained within the modules:

* `tcga.pattern`: This module contains the actual implementation of the 
  pattern recovery algorithm.  Its only function is
  `tcga.pattern.dag_pattern_recover()`.
* `tcga.compare`: Contains objective functions and implementations of the
  boolean functions.

### Utilities

The previous set of packages leaves much to be desired in terms of usability.
The next group of packages simplify working with the framework.

* `tcga.parse`: This module, at a minimum, simplifies parsing data stored as 
  files.  When used to its full extent, the module creates an entire 
  framework for working with Python data on the filesystem.  The benefits of 
  the approach are speed (by 'caching' parsed data as pickles) and brevity 
  (refer to data on the filesystem by a title, rather than full file path).
  See its documentation for more details.
* `tcga.tree`: This module provides convenient functions for examining the 
DAG, after it has been modified by `dag_pattern_recover()`.

### Experiments

The `smbio` library provides a multiprocessing base class for running
experiments.  This package contains a few modules that extend this class, in
order to perform permutation tests.

Info
----

* Author: Stephen Brennan <stephen.brennan@case.edu>
* Advisors: Mehmet Koyuturk, Matthew Ruffalo
