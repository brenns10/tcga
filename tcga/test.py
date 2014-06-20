#-------------------------------------------------------------------------------
# 
# File:         test.py
#
# Author:       Stephen Brennan
#
# Date Created: Tuesday, 10 June 2014
#
# Description:  Unit tests for TCGA package.
#
#-------------------------------------------------------------------------------

import unittest

import numpy as np
from pandas import Series

# Hackish import system!  Run tests anywhere!
try:
    # Relative package imports are used when running python -m unittest from
    # the repository root.
    from . import compare

except SystemError:
    try:
        # Absolute package imports are used when running unit tests from
        # PyCharm IDE.
        from tcga import compare

    except ImportError:
        # Straight module imports are used when running:
        #   python -m unittest from the inner tcga directory
        #   python tcga/test.py
        #   python test.py
        import compare


class TestMutualInfo(unittest.TestCase):

    def test_trivial_one_case(self):
        """Tests the trivial case when one dataset is determined by another."""
        basis = Series([0, 0, 1, 1]).astype(bool)
        opposite = Series([1, 1, 0, 0]).astype(bool)

        self.assertEqual(compare.binary_mutual_information(basis, basis), 1)
        self.assertEqual(compare.binary_mutual_information(basis, opposite), 1)

    def test_trivial_zero_case(self):
        """Tests the trivial case when there is no mutual information."""
        basis = Series([0, 0, 1, 1]).astype(bool)
        no_mutual_info = Series([0, 1, 0, 1]).astype(bool)
        self.assertEqual(compare.binary_mutual_information(basis,
                                                           no_mutual_info), 0)

    def _get_binary_dataset(self, size, index):
        """
        Creates a binary dataset from an "index".

        It's basically a list of booleans from the binary representation of
        the index.

        :param size: Number of bools in the dataset.
        :param index: The index of the dataset (from 0 to 2^size - 1).
        """
        ds = [False] * size
        for idx in range(size):
            ds[idx] = index % 2 == 1
            index //= 2
        return Series(ds)

    def test_compare_methods(self):
        """
        Tests all mutual information methods to ensure uniformity.

        For all combinations of X-length binary datasets, ensures that all
        five mutual information functions return similar results.
        """
        size = 5
        for x in range(2**size):
            ds1 = self._get_binary_dataset(size, x)
            for y in range(x, 2**size):
                ds2 = self._get_binary_dataset(size, y)
                self.assertTrue(compare.compare_mi_methods(ds1, ds2),
                                ''.join(['X:', str(x), ', Y:', str(y)]))


class TestEntropy(unittest.TestCase):

    def test_trivial_zero_case(self):
        """Tests homogeneous data where entropy is zero."""
        first = Series([True, True, True, True])
        second = Series([False, False, False, False])
        
        self.assertEqual(compare.entropy(first), 0)
        self.assertEqual(compare.entropy(second), 0)

    def test_trivial_one_case(self):
        """Tests uniformly distributed data where entropy is one."""
        data = Series([True, True, False, False])
        self.assertEqual(compare.entropy(data), 1)

    def test_basic_case(self):
        """Tests a basic case where the entropy is not zero or one."""
        dataset1 = Series([True, True, True, False])
        dataset2 = Series([False, False, False, True])
        # H(data) = -.25*log_2(.25) - 0.75*log_2(.75)
        # H(data) = 0.5 + 0.31127812445913283
        # H(data) = 0.81127812445913283
        self.assertTrue(np.isclose([compare.entropy(dataset1)],
                                   [0.81127812445913283]))
        self.assertTrue(np.isclose([compare.entropy(dataset2)],
                                   [0.81127812445913283]))


class TestBestCombination(unittest.TestCase):

    def test_simple(self):
        """Tests that best_combination() can identify a complete pattern."""
        ds1 = Series([False, False, True, True])
        ds2 = Series([False, True, False, True])

        for logic_func in compare.COMBINATIONS:
            phenotype = logic_func(ds1, ds2)
            best_func, *etc = compare.best_combination(ds1, ds2, phenotype)
            self.assertEqual(logic_func, best_func)

if __name__ == '__main__':
    unittest.main()
