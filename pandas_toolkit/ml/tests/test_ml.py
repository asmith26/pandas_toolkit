import unittest

import numpy as np
import pandas as pd

import pandas_toolkit.ml


class TestStandardScaler(unittest.TestCase):
    def test_standard_scaler_accessor_usage(self):
        df = pd.DataFrame({"x": [0, 1], "y": [0, 1]}, index=[0, 1])

        actual_s = df.ml.standard_scaler(column="x")

        expected_s = pd.Series([-1, 1])

        pd.testing.assert_series_equal(expected_s, actual_s, check_exact=True)


class TestTrainValidationSplit(unittest.TestCase):
    def test_with_sensible_is_validation_frac(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])

        np.random.seed(1)
        actual_s = df.ml.train_validation_split(is_validation_frac=2 / 3)

        expected_s = pd.Series([1, 0, 1], dtype=np.int8)

        pd.testing.assert_series_equal(expected_s, actual_s, check_exact=True)

    def test_with_not_perfectly_divisible_is_validation_frac(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])

        np.random.seed(1)
        actual_s = df.ml.train_validation_split(is_validation_frac=0.5)

        expected_s = pd.Series([1, 0, 0], dtype=np.int8)

        pd.testing.assert_series_equal(expected_s, actual_s, check_exact=True)

    def test_with_is_validation_frac_eq_1(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])

        actual_s = df.ml.train_validation_split(is_validation_frac=1)

        expected_s = pd.Series([1, 1, 1], dtype=np.int8)

        pd.testing.assert_series_equal(expected_s, actual_s, check_exact=True)

    def test_with_is_validation_frac_eq_0(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])

        actual_s = df.ml.train_validation_split(is_validation_frac=0)

        expected_s = pd.Series([0, 0, 0], dtype=np.int8)

        pd.testing.assert_series_equal(expected_s, actual_s, check_exact=True)
