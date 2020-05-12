from typing import Tuple

import pandas as pd


@pd.api.extensions.register_dataframe_accessor("ml")
class MachineLearningAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def train_test_split(self, train_frac: float, random_seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :param train_frac: Fraction of the total number of rows to be included in the df_train output.
        :param random_seed: Seed for the random number generator (e.g. for reproducible splits).
        :return: A tuple of 2 distinct random samples of the original dataframe.
        """
        df_train = self._df.sample(frac=train_frac, random_state=random_seed)
        df_test = self._df.drop(df_train.index)
        return df_train, df_test
