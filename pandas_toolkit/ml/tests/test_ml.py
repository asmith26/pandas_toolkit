import unittest

import pandas as pd

import pandas_toolkit.ml


class TestStandardScaler(unittest.TestCase):
    def test_standard_scaler_accessor_usage(self):
        df = pd.DataFrame({"x": [0, 1], "y": [0, 1]}, index=[0, 1])

        actual_s = df.ml.standard_scaler(column="x")
        actual_sklearn_transform: pandas_toolkit.ml.MLTransform = df.ml.transforms["standard_scaler"]

        pd.testing.assert_series_equal(pd.Series([-1, 1]), actual_s, check_exact=True)
        self.assertEqual("x", actual_sklearn_transform.column_name)
        self.assertEqual(0.25, actual_sklearn_transform.sklearn_object.var_[0])
        self.assertEqual(0.5, actual_sklearn_transform.sklearn_object.mean_[0])


class TestApplyDfTrainTransform(unittest.TestCase):
    def test_returns_correctly_when_standard_scaler_used(self):
        df_train = pd.DataFrame({"x": [0, 1], "y": [0, 1]}, index=[0, 1])
        df_validation = pd.DataFrame({"x": [2], "y": [2]}, index=[0])

        _ = df_train.ml.standard_scaler(column="x")  # tested elsewhere
        actual_s_validation = df_validation.ml.apply_df_train_transform(df_train.ml.transforms["standard_scaler"])

        pd.testing.assert_series_equal(pd.Series([3]), actual_s_validation, check_exact=True)


class TestTrainValidationSplit(unittest.TestCase):
    def test_with_sensible_is_validation_frac(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])
        actual_df_train, actual_df_validation = df.ml.train_validation_split(train_frac=2 / 3, random_seed=42)

        expected_df_train = pd.DataFrame({"x": [0, 1], "y": [0, 1]}, index=[0, 1])
        expected_df_validation = pd.DataFrame({"x": [2], "y": [2]}, index=[2])

        pd.testing.assert_frame_equal(expected_df_train, actual_df_train, check_exact=True)
        pd.testing.assert_frame_equal(actual_df_validation, expected_df_validation, check_exact=True)

    def test_with_not_perfectly_divisible_is_validation_frac(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])
        actual_df_train, actual_df_validation = df.ml.train_validation_split(train_frac=0.5, random_seed=42)

        expected_df_train = pd.DataFrame({"x": [0, 1], "y": [0, 1]}, index=[0, 1])
        expected_df_validation = pd.DataFrame({"x": [2], "y": [2]}, index=[2])

        pd.testing.assert_frame_equal(expected_df_train, actual_df_train, check_exact=True)
        pd.testing.assert_frame_equal(actual_df_validation, expected_df_validation, check_exact=True)

    def test_with_is_validation_frac_eq_1(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])
        actual_df_train, actual_df_validation = df.ml.train_validation_split(train_frac=1, random_seed=42)

        expected_df_train = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])
        expected_df_validation = pd.DataFrame({"x": [], "y": []}, dtype="int64")

        pd.testing.assert_frame_equal(expected_df_train, actual_df_train, check_exact=True)
        pd.testing.assert_frame_equal(actual_df_validation, expected_df_validation, check_exact=True)

    def test_with_is_validation_frac_eq_0(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])
        actual_df_train, actual_df_validation = df.ml.train_validation_split(train_frac=0, random_seed=42)

        expected_df_train = pd.DataFrame({"x": [], "y": []}, dtype="int64")
        expected_df_validation = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]}, index=[0, 1, 2])

        pd.testing.assert_frame_equal(expected_df_train, actual_df_train, check_exact=True)
        pd.testing.assert_frame_equal(actual_df_validation, expected_df_validation, check_exact=True)
