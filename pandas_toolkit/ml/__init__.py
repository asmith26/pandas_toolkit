import numpy as np
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("ml")
class MachineLearningAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def train_test_split(self, is_train_prob: float) -> pd.Series:
        """
        **Parameters**
        > **is_train_prob:**  Probability of row being marked as 1 (i.e. is_train = True).

        **Returns**
        > A pd.Series with values 0 and 1 randomly selected with probability 1-is_train_prob and is_train_prob, respectively.

        Examples
        ```python
        >>> df = pd.DataFrame({"x": [0, 1, 2],
                               "y": [0, 1, 2]},
                               index=[0, 1, 2])
        >>> df["is_train"] = df.ml.train_test_split(train_frac=2/3)
        >>> df["is_train"]
        pd.Series([0, 1, 1])
        ```
        """
        arr_is_train = np.random.choice([0, 1], size=len(self._df), p=[1 - is_train_prob, is_train_prob])
        s_is_train = pd.Series(arr_is_train, self._df.index)
        return s_is_train
