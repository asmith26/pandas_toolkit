import unittest

import pandas as pd

import pandas_toolkit.ml


class TestMachineLearningAccessor(unittest.TestCase):
    def test_train_test_split(self):
        df = pd.DataFrame({
            "x": [0, 1, 2],
            "y": [0, 1, 2],
        }, index=[0, 1, 2])
        actual_df_train, actual_df_test = df.ml.train_test_split(train_frac=2/3, random_seed=42)

        expected_df_train = pd.DataFrame({
            "x": [0, 1],
            "y": [0, 1],
        }, index=[0, 1])
        expected_df_test = pd.DataFrame({
            "x": [2],
            "y": [2],
        }, index=[2])

        pd.testing.assert_frame_equal(expected_df_train, actual_df_train, check_exact=True)
        pd.testing.assert_frame_equal(expected_df_test, actual_df_test, check_exact=True)


if __name__ == '__main__':
    unittest.main()
