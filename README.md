TCGA Data Analysis Package
==========================

Summary
-------

This package implements an algorithm which uses existing biological knowledge
to assist in determining the best binary function to explain a phenotype 
variable, using somatic mutation data.  The existing biological knowledge is 
presented in a binary DAG form of a gene ontology.  

The package allows for using a variety of phenotypes and objective functions 
to guide the search.  The currently implemented choices are:

* Patient lifetime as a phenotype, using the log rank test for the objective 
  function.
* Any binary phenotype (e.g. vital status), using mutual information as the 
  objective function.
  
More phenotypes and objective functions may be added readily, 
as discussed below.

Project Technologies and Dependencies
-------------------------------------

* Python 3
* [Pandas](https://pypi.python.org/pypi/pandas)
* [NumPy](https://pypi.python.org/pypi/numpy)
* [SymPy](https://pypi.python.org/pypi/sympy) (to present symbolic 
  representations of binary functions)
* [NetworkX](https://pypi.python.org/pypi/networkx) (to store the binary DAG)
* [Lifelines](https://pypi.python.org/pypi/lifelines) (for log rank test)

Structure
---------

### Framework

The framework of the pattern recovery algorithm is contained within the modules:

* `tcga.pattern`: This module contains the actual implementation of the 
  pattern recovery algorithm.  Its only function is
  `tcga.pattern.dag_pattern_recover()`.
* `tcga.compare`: Contains objective functions for the pattern recovery 
  algorithm.  New objective functions should be modeled after those in this 
  package.

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

In order to learn about the algorithm, and to use it on real data, 
an experiment abstraction is provided.  The abstraction allows for tasks to 
be repeated over many configurations.  Experiments created using this 
abstraction may be run concurrently using the Python standard 
library `multiprocessing` module, allowing for huge performance gains on 
multi-core systems.

The `Experiment` abstraction is provided in the `experiment` module.  
Existing experiments are implemented in `detection_experiment` and 
`permutation_test`.

### Miscellaneous

Before the existence of the Experiment abstraction, this module had a command
line interface (`run.py`) to streamline running tests.  However, 
experiments are most easily run within an IPython shell.  The CLI may be 
removed in the future (it is empty currently).

The CLI made heavy use of utilities in the `util` module.  Many of them are 
useful for creating the actual command line interface, and have nothing to do
with the pattern recovery algorithm itself.  These utilities may also be 
removed, or extracted into their own library.