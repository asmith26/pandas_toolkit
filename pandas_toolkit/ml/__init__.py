from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SklearnObject(object):
    @staticmethod
    def transform(x: np.ndarray) -> np.ndarray:  # pragma: no cover
        pass


class MLTransform(object):
    def __init__(self, column_name: str, sklearn_object: SklearnObject):
        self.column_name = column_name
        self.sklearn_object = sklearn_object


@pd.api.extensions.register_dataframe_accessor("ml")
class MachineLearningAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.transforms: Dict[str, MLTransform] = {}  # format: {"transform name": MLTransform}

    def apply_df_train_transform(self, ml_transform: MLTransform) -> pd.Series:
        """
        **Parameters**
        > **ml_transform:** `pandas_toolkit.ml.MLTransform` object containing transform to apply and column name to
          be applied to (normally via e.g. `df_train.ml.transforms["standard_scaler"]`).

        **Returns**
        > Transformed featured using e.g. `df_train` data statistics.

        Examples
        ```python
        >>> df_train = pd.DataFrame({"x": [0, 1],
                                     "y": [0, 1]},
                                     index=[0, 1])
        >>> df_validation = pd.DataFrame({"x": [2],
                                          "y": [2]},
                                          index=[0])
        >>> df_train["standard_scaler_x"] = df_train.ml.standard_scaler(column="x")
        >>> df_train["standard_scaler_x"]
        pd.Series([-1, 1])

        >>> df_train.ml.transforms
        {'standard_scaler': <pandas_toolkit.ml.MLTransform object at 0x7f1af20f0af0>}

        >>> df_validation["standard_scaler_x"] = \\
                df_validation.ml.apply_df_train_transform(df_train.ml.transforms["standard_scaler"])
        >>> df_validation["standard_scaler_x"]
        pd.Series([3])
        ```
        """
        column = ml_transform.column_name
        sklearn_object = ml_transform.sklearn_object

        s = self._df[column]
        arr_transformed_col: np.ndarray = sklearn_object.transform(s.values.reshape(-1, 1))
        s_transformed_col = pd.Series(data=arr_transformed_col.flatten(), index=self._df.index, dtype=s.dtype)
        return s_transformed_col

    def standard_scaler(self, column: str) -> pd.Series:
        """
        **Parameters**
        > **column:** Column denoting feature to standardize.

        **Returns**
        > Standardized featured by removing the mean and scaling to unit variance (via
          [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)):
          `z = (x - u) / s` (`u` := mean of training samples, `s` := standard deviation of training samples).

        **Side Effects**
        > Updates the `df.ml.transforms` dictionary with key "standard_scaler" and value
          `pandas_toolkit.ml.MLTransform` corresponding to the column name and fitted
          `sklearn.preprocessing.StandardScaler` object.

        Examples
        ```python
        >>> df = pd.DataFrame({"x": [0, 1],
                               "y": [0, 1]},
                               index=[0, 1])
        >>> df["standard_scaler_x"] = df.ml.standard_scaler(column="x")
        >>> df["standard_scaler_x"]
        pd.Series([-1, 1])

        >>> df.ml.transforms
        {'standard_scaler': <pandas_toolkit.ml.MLTransform object at 0x7f1af20f0af0>}
        ```
        """
        s = self._df[column]
        scaler = StandardScaler()
        arr_scaled_col: np.ndarray = scaler.fit_transform(s.values.reshape(-1, 1))
        s_scaled_col = pd.Series(data=arr_scaled_col.flatten(), index=self._df.index, dtype=s.dtype)

        self.transforms["standard_scaler"] = MLTransform(column_name=column, sklearn_object=scaler)
        return s_scaled_col

    def train_validation_split(self, train_frac: float, random_seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        **Parameters**
        > **train_frac:** Fraction of rows to be added to df_train.

        > **random_seed:** Seed for the random number generator (e.g. for reproducible splits).

        **Returns**
        > `df_train` and `df_validation`, split from the original dataframe.

        Examples
        ```python
        >>> df = pd.DataFrame({"x": [0, 1, 2],
                               "y": [0, 1, 2]},
                               index=[0, 1, 2])
        >>> df_train, df_validation = df.ml.train_validation_split(train_frac=2/3)
        >>> df_train
        pd.DataFrame({"x": [0, 1], "y": [0, 1]}, index=[0, 1]),
        >>> df_validation
        pd.DataFrame({"x": [2], "y": [2]}, index=[2])
        ```
        """
        df_train = self._df.sample(frac=train_frac, random_state=random_seed)
        df_validation = self._df.drop(labels=df_train.index)
        return df_train, df_validation
