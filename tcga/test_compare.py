"""Unit tests for tcga.compare."""

import unittest

import numpy as np
from pandas import Series

from tcga import compare


class TestEntropy(unittest.TestCase):

    def test_trivial_zero_case(self):
        """Tests homogeneous data where _entropy is zero."""
        first = Series([True, True, True, True])
        second = Series([False, False, False, False])

        self.assertEqual(compare._entropy(first), 0)
        self.assertEqual(compare._entropy(second), 0)

    def test_trivial_one_case(self):
        """Tests uniformly distributed data where _entropy is one."""
        data = Series([True, True, False, False])
        self.assertEqual(compare._entropy(data), 1)

    def test_basic_case(self):
        """Tests a basic case where the _entropy is not zero or one."""
        dataset1 = Series([True, True, True, False])
        dataset2 = Series([False, False, False, True])
        # H(data) = -.25*log_2(.25) - 0.75*log_2(.75)
        # H(data) = 0.5 + 0.31127812445913283
        # H(data) = 0.81127812445913283
        self.assertTrue(np.isclose([compare._entropy(dataset1)],
                                   [0.81127812445913283]))
        self.assertTrue(np.isclose([compare._entropy(dataset2)],
                                   [0.81127812445913283]))


class TestMutualInfo(unittest.TestCase):

    def test_trivial_one_case(self):
        """Tests the trivial case when one dataset is determined by another."""
        basis = Series([0, 0, 1, 1]).astype(bool)
        opposite = Series([1, 1, 0, 0]).astype(bool)

        self.assertEqual(compare.mutual_info(basis, basis), 1)
        self.assertEqual(compare.mutual_info(basis, opposite), 1)

    def test_trivial_zero_case(self):
        """Tests the trivial case when there is no mutual information."""
        basis = Series([0, 0, 1, 1]).astype(bool)
        no_mutual_info = Series([0, 1, 0, 1]).astype(bool)
        self.assertEqual(compare.mutual_info(basis, no_mutual_info), 0)

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


if __name__ == '__main__':
    unittest.main()
