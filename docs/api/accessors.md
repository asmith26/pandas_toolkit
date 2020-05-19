# Accessors API

## ML Methods
#### `standard_scaler` *<small>[[source](https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/ml/__init__.py#L11)]</small>*
`standard_scaler`*(self, column: str) -> pd.Series*

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

#### `train_test_split` *<small>[[source](https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/ml/__init__.py#L35)]</small>*
`train_test_split`*(self, is_train_frac: float) -> pd.Series*

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

