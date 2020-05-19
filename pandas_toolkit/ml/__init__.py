import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@pd.api.extensions.register_dataframe_accessor("ml")
class MachineLearningAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def standard_scaler(self, column: str) -> pd.Series:
        """
        **Parameters**
        > **column:**  Column denoting feature to standardize..

        **Returns**
        > Standardized featured by removing the mean and scaling to unit variance: `z = (x - u) / s`

        Examples
        ```python
        >>> df = pd.DataFrame({"x": [0, 1],
                               "y": [0, 1]},
                               index=[0, 1])
        >>> df["standard_scaler_x"] = df.ml.standard_scaler(column="x")
        >>> df["standard_scaler_x"]
        pd.Series([-1, 1])
        ```
        """
        s = self._df[column]
        scaler = StandardScaler()
        arr_scaled_col: np.ndarray = scaler.fit_transform(s.values.reshape(-1, 1))
        s_scaled_col = pd.Series(data=arr_scaled_col.flatten(), index=self._df.index, dtype=s.dtype)
        return s_scaled_col

    def train_test_split(self, is_train_frac: float) -> pd.Series:
        """
        **Parameters**
        > **is_train_frac:**  Fraction of row being marked as 1 (i.e. is_train = True).

        **Returns**
        > A pd.Series with values 0 and 1 randomly selected with fraction 1-is_train_frac and is_train_frac,
          respectively.

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

        s_is_train = pd.Series(data=arr_is_train, index=self._df.index, dtype=np.int8)
        return s_is_train
