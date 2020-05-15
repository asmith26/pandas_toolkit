# Accessors API

## ML Methods
#### `train_test_split` *<small>[[source](https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/ml/__init__.py#L11)]</small>*
`train_test_split`*(self, train_frac: float, random_seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]*

**Parameters**
> **train_frac:**  Fraction of the total number of rows to be included in the df_train output.
> **random_seed:** Seed for the random number generator (e.g. for reproducible splits).

**Returns**
> A tuple of 2 distinct random samples of the original dataframe.

Examples
```python
>>> df = pd.DataFrame({"x": [0, 1, 2],
                       "y": [0, 1, 2]},
                       index=[0, 1, 2])
>>> actual_df_train, actual_df_test = df.ml.train_test_split(train_frac=2/3, random_seed=0)
>>> actual_df_train
pd.DataFrame({"x": [2, 1], "y": [2, 1]}, index=[2, 1])
>>> actual_df_test
pd.DataFrame({"x": [0], "y": [0]}, index=[0])
```