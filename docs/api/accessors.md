# Accessors API

## df.ml. Methods
---
### `apply_df_train_transform`
`apply_df_train_transform`*(<span style='color:green'>ml_transform</span>: <span style='color:blue'>MLTransform</span>) -> pd.Series* *<small>[[source]](https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/ml/__init__.py#L26)</small>*

**Parameters**
> **ml_transform:** `pandas_toolkit.ml.MLTransform` object containing transform to apply and column name to
  apply it (normally via e.g. `df_train.ml.transforms["standard_scaler"]`).

**Returns**
> Transformed featured using e.g. df_train data statistics.

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

>>> df_validation["standard_scaler_x"] = \
        df_validation.ml.apply_df_train_transform(df_train.ml.transforms["standard_scaler"])
>>> df_validation["standard_scaler_x"]
pd.Series([3])
```
---
### `standard_scaler`
`standard_scaler`*(<span style='color:green'>column</span>: <span style='color:blue'>str</span>) -> pd.Series* *<small>[[source]](https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/ml/__init__.py#L64)</small>*

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
---
### `train_validation_split`
`train_validation_split`*(<span style='color:green'>train_frac</span>: <span style='color:blue'>float</span>, <span style='color:green'>random_seed</span>: <span style='color:blue'>int = None</span>) -> Tuple[pd.DataFrame, pd.DataFrame]* *<small>[[source]](https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/ml/__init__.py#L100)</small>*

**Parameters**
> **train_frac:** Fraction of rows to be added to df_train.

> **random_seed:** Seed for the random number generator (e.g. for reproducible splits).

**Returns**
> df_train and df_validation, split from the original dataframe.

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
---
## df.nn. Methods
---
### `init`
`init`*(<span style='color:green'>x_columns</span>: <span style='color:blue'>List[str]</span>, <span style='color:green'>y_columns</span>: <span style='color:blue'>List[str]</span>, <span style='color:green'>net_function</span>: <span style='color:blue'>Callable[[jnp.ndarray] jnp.ndarray]</span>, <span style='color:green'>loss</span>: <span style='color:blue'>str</span>, <span style='color:green'>optimizer</span>: <span style='color:blue'>InitUpdate = optix.adam(learning_rate=1e-3)</span>, <span style='color:green'>batch_size</span>: <span style='color:blue'>int = None</span>, <span style='color:green'>apply_rng</span>: <span style='color:blue'>jnp.ndarray = None</span>) -> pd.DataFrame* *<small>[[source]](https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/nn/__init__.py#L38)</small>*

**Parameters**
> **x_columns:** Columns to be used as input for the model.

> **y_columns:** Columns to be used as output for the model.

> **net_function:** A function that defines a haiku.Sequential neural network and how to predict uses it (this
function is passed to hk.transform). This should have the signature *net_function(x: jnp.ndarray) ->
jnp.ndarray*.

> **loss:** Loss function to use. See available loss functions in
[jax_toolkit](https://asmith26.github.io/jax_toolkit/losses_and_metrics/).

> **optimizer:** Optimizer to use. See [jax](https://jax.readthedocs.io/en/latest/jax.experimental.optix.html).

> **batch_size:** Batch size to use. If not specified, the number of rows in the entire dataframe is used.

> **apply_rng:** If your net_function is non-deterministic, set this value to some `jax.random.PRNGKey(seed)`
 for repeatable outputs.

**Returns**
> A pd.DataFrame containing a neural network model ready for training with pandas_toolkit.

Examples
```python
>>> def net_function(x: jnp.ndarray) -> jnp.ndarray:
...     net = hk.Sequential([relu])
...     predictions: jnp.ndarray = net(x)
...     return predictions
>>> df_train = df_train.nn.init(x_columns=["x"],
...                             y_columns=["y"],
...                             net_function=net_function,
...                             loss="mean_squared_error")
>>> for _ in range(10):  # num_epochs
...     df_train = df_train.nn.update(df_validation_to_plot=df_validation)
```
---
### `get_model`
`get_model`*() -> Model* *<small>[[source]](https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/nn/__init__.py#L98)</small>*

**Returns**
> A pandas_toolkit.nn.Model object. As this is not linked to a pd.DataFrame, it is much more lightweight
and could be used in e.g. a production setting.

Examples
```python
>>> model = df_train.nn.get_model()
>>> model.predict(x=jnp.array([42]))
```
---
### `hvplot_losses`
`hvplot_losses`*() -> None* *<small>[[source]](https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/nn/__init__.py#L112)</small>*

**Returns**
> A Holoviews object for interactive (via Bokeh), real-time ploting of training and validation loss
curves. For an example usage, see [this notebook](
https://github.com/asmith26/pandas_toolkit/blob/master/notebooks/sine.ipynb).

Examples
```python
>>> df_train.nn.hvplot_losses()
```
---
### `update`
`update`*(<span style='color:green'>df_validation_to_plot</span>: <span style='color:blue'>pd.DataFrame = None</span>) -> pd.DataFrame* *<small>[[source]](https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/nn/__init__.py#L141)</small>*

**Parameters**
> **df_validation_to_plot:** Validation data to evaluate and update loss curve with.

**Returns**
> A pd.DataFrame containing an updated neural network model (trained on one extra epoch).

Examples
```python
>>> for _ in range(10):  # num_epochs
...     df_train = df_train.nn.update(df_validation_to_plot=df_validation)
```
---
### `predict`
`predict`*(<span style='color:green'>x_columns</span>: <span style='color:blue'>List[str] = None</span>) -> pd.Series* *<small>[[source]](https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/nn/__init__.py#L174)</small>*

**Parameters**
> **x_columns:** Columns to predict on. If `None`, the same x_columns names used to train the model are used.

**Returns**
> A pd.Series of predictions.

Examples
```python
>>> df_new = pd.DataFrame({"x": [-10, -5, 22]})
>>> df_new.model = df_train.nn.get_model()
>>> df_new["predictions"] = df_new.nn.predict()
```
---
### `evaluate`
`evaluate`*(<span style='color:green'>x_columns</span>: <span style='color:blue'>List[str] = None</span>, <span style='color:green'>y_columns</span>: <span style='color:blue'>List[str] = None</span>) -> pd.Series* *<small>[[source]](https://github.com/asmith26/pandas_toolkit/blob/master/pandas_toolkit/nn/__init__.py#L195)</small>*

**Parameters**
> **x_columns:** Columns to predict on. If `None`, the same x_columns names used to train the model are used.

> **y_columns:** Columns with true output values to compare predicted values with. If `None`, the same
y_columns names used to train the model are used.

**Returns**
> Evaluation of the prediction using the loss_function provided in `df.nn.init(...)`.

Examples
```python
>>> df_test = pd.DataFrame({"x": [-1, 0, 1], "y": [0, 0, 1]})
>>> df_test.model = df_train.nn.get_model()
>>> df_test.nn.evaluate()
```
---
