import numpy as np
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("ml")
class MachineLearningAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def train_test_split(self, is_train_frac: float) -> pd.Series:
        """
        **Parameters**
        > **is_train_frac:**  Fraction of row being marked as 1 (i.e. is_train = True).

        **Returns**
        > A pd.Series with values 0 and 1 randomly selected with fraction 1-is_train_frac and is_train_frac, respectively.

        Examples
        ```python
        >>> df = pd.DataFrame({"x": [0, 1, 2],
                               "y": [0, 1, 2]},
                               index=[0, 1, 2])
        >>> df["is_train"] = df.ml.train_test_split(is_train_frac=2/3)
        >>> df["is_train"]
        pd.Series([0, 1, 1])
        ```
        """
        num_rows = len(self._df)
        num_train = int(num_rows * is_train_frac)
        num_test = num_rows - num_train

        arr_is_train = np.concatenate([np.ones(num_train), np.zeros(num_test)])
        np.random.shuffle(arr_is_train)

        s_is_train = pd.Series(arr_is_train, self._df.index, dtype=np.int8)
        return s_is_train
