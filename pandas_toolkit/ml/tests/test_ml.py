import unittest

import numpy as np
import pandas as pd

import pandas_toolkit.ml


class TestMachineLearningAccessor(unittest.TestCase):
    def test_train_test_split_with_sensible_train_frac(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])

        np.random.seed(1)
        actual_s = df.ml.train_test_split(is_train_prob=2 / 3)

        expected_s = pd.Series([1, 1, 0])

        pd.testing.assert_series_equal(expected_s, actual_s, check_exact=True)

    def test_train_test_split_with_train_frac_eq_1(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])

        actual_s = df.ml.train_test_split(is_train_prob=1)

        expected_s = pd.Series([1, 1, 1])

        pd.testing.assert_series_equal(expected_s, actual_s, check_exact=True)

    def test_train_test_split_with_train_frac_eq_0(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])

        actual_s = df.ml.train_test_split(is_train_prob=0)

        expected_s = pd.Series([0, 0, 0])

        pd.testing.assert_series_equal(expected_s, actual_s, check_exact=True)
