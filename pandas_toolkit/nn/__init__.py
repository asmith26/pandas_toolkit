import math
import random
from typing import Callable, List, Optional

import pandas as pd
from jax import numpy as jnp
from jax.experimental import optix
from jax.experimental.optix import InitUpdate

from pandas_toolkit.nn.Model import Model
from pandas_toolkit.utils.custom_types import Batch


def _get_num_batches(num_rows: int, batch_size: Optional[int]) -> int:
    if batch_size is None:
        return 1
    num_batches = num_rows / batch_size
    if math.ceil(num_batches) == math.floor(num_batches):
        return int(num_batches)
    return int(num_batches) + 1


def _get_batch(
    df: pd.DataFrame, batch_number: int, batch_size: int, x_columns: List[str], y_columns: Optional[List[str]]
) -> Batch:
    start_batch_idx = batch_number * batch_size
    end_batch_idx = (batch_number + 1) * batch_size

    df_train_batch = df.iloc[start_batch_idx:end_batch_idx]
    if y_columns is None:
        return Batch(x=jnp.array(df_train_batch.loc[:, x_columns].values), y=None)
    return Batch(
        x=jnp.array(df_train_batch.loc[:, x_columns].values), y=jnp.array(df_train_batch.loc[:, y_columns].values)
    )


@pd.api.extensions.register_dataframe_accessor("nn")
class NeuralNetworkAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df = df

        self._num_rows = len(df)

    def init(
        self,
        x_columns: List[str],
        y_columns: List[str],
        net_function: Callable[[jnp.ndarray], jnp.ndarray],
        loss: str,
        optimizer: InitUpdate = optix.adam(learning_rate=1e-3),
        batch_size: int = None,
        apply_rng: bool = False,
        rng_seed: int = random.randint(0, int(1e15)),  # nosec
    ) -> pd.DataFrame:
        """
        **Parameters**
        > **x_columns:** Columns to be used as input for the model.

        > **y_columns:** Columns to be used as output for the model.

        > **net_function:** A function that defines a dm-haiku neural network and how to predict uses it (this
        function is passed to hk.transform). This should have the signature `net_function(x: jnp.ndarray) ->
        jnp.ndarray`.

        > **loss:** Loss function to use. See available loss functions in
        [jax_toolkit](https://asmith26.github.io/jax_toolkit/losses_and_metrics/).

        > **optimizer:** Optimizer to use. See [jax](https://jax.readthedocs.io/en/latest/jax.experimental.optix.html).

        > **batch_size:** Batch size to use. If not specified, the number of rows in the entire dataframe is used.

        > **apply_rng:** If your `net_function` is non-deterministic, set this value to `True` to enable you model to
        predict with randomness.

        > **rng_seed:** Set a seed for reprodubility.

        **Returns**
        > `df` containing a neural network model ready for training with pandas_toolkit.

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
        """
        self._num_batches = _get_num_batches(num_rows=self._num_rows, batch_size=batch_size)
        self._batch_size = batch_size if batch_size is not None else self._num_rows

        num_features = len(x_columns)
        example_x = jnp.zeros(shape=[self._batch_size, num_features])
        self._df.model = Model(net_function, loss, optimizer, example_x, apply_rng, rng_seed)

        self._df.model._x_columns = x_columns
        self._df.model._y_columns = y_columns

        return self._df

    def get_model(self) -> Model:
        """
        **Returns**
        > A `pandas_toolkit.nn.Model` object. As this is not linked to a `pd.DataFrame`, it is much more lightweight
        and could be used in e.g. a production setting.

        Examples
        ```python
        >>> model = df_train.nn.get_model()
        >>> model.predict(x=jnp.array([42]))
        ```
        """
        return self._df.model.copy()  # type: ignore

    def hvplot_losses(self):  # type: ignore  # pragma: no cover
        """
        **Returns**
        > A Holoviews object for interactive (via Bokeh), real-time ploting of training and validation loss
        curves. For an example usage, see [this notebook](
        https://github.com/asmith26/pandas_toolkit/blob/master/notebooks/sine.ipynb).

        Examples
        ```python
        >>> df_train.nn.hvplot_losses()
        ```
        """
        from streamz import Stream
        from streamz.dataframe import DataFrame

        self.sdf = DataFrame(Stream(), example=pd.DataFrame({"epoch": [], "train_loss": [], "validation_loss": []}))
        return self.sdf.hvplot.line(x="epoch", y=["train_loss", "validation_loss"])

    def _update_hvplot_losses(self, df_validation: pd.DataFrame) -> None:  # pragma: no cover
        df_validation.model = self.get_model()
        df_losses = pd.DataFrame(
            {
                "epoch": [self._df.model.num_epochs],
                "train_loss": self.evaluate(batch_size=self._batch_size).tolist(),
                "validation_loss": df_validation.nn.evaluate(batch_size=self._batch_size).tolist(),
            }
        )
        self.sdf.emit(df_losses)

    def update(self, df_validation_to_plot: pd.DataFrame = None) -> pd.DataFrame:
        """
        **Parameters**
        > **df_validation_to_plot:** Validation data to evaluate and update loss curve with.

        **Returns**
        > `df` containing an updated neural network model (trained on one extra epoch).

        Examples
        ```python
        >>> for _ in range(10):  # num_epochs
        ...     df_train = df_train.nn.update(df_validation_to_plot=df_validation)
        ```
        """
        df_train = self._df.sample(frac=1)

        for batch_number in range(self._num_batches):
            batch = _get_batch(
                df_train, batch_number, self._batch_size, self._df.model._x_columns, self._df.model._y_columns
            )
            self._df.model._update(batch.x, batch.y)

        self._df.model.num_epochs += 1

        if df_validation_to_plot is not None:
            self._update_hvplot_losses(df_validation_to_plot)  # pragma: no cover

        return self._df

    def predict(self, x_columns: List[str] = None, batch_size: int = None) -> jnp.ndarray:
        """
        **Parameters**
        > **x_columns:** Columns to predict on. If `None`, the same `x_columns` names used to train the model are used.

        > **batch_size:** Batch size to use. If not specified, the entire dataset is used.

        **Returns**
        > A pd.Series of predictions.

        Examples
        ```python
        >>> df_new = pd.DataFrame({"x": [-10, -5, 22]})
        >>> df_new.model = df_train.nn.get_model()
        >>> df_new["predictions"] = df_new.nn.predict()
        ```
        """
        if x_columns is None:
            x_columns = self._df.model._x_columns
        batch_size = batch_size if batch_size is not None else self._num_rows
        num_batches = _get_num_batches(num_rows=self._num_rows, batch_size=batch_size)

        predictions = []
        for batch_number in range(num_batches):
            batch = _get_batch(self._df, batch_number, batch_size, x_columns, None)
            predictions.append(self._df.model.predict(batch.x))

        return jnp.vstack(predictions)

    def evaluate(self, x_columns: List[str] = None, y_columns: List[str] = None, batch_size: int = None) -> jnp.ndarray:
        """
        **Parameters**
        > **x_columns:** Columns to predict on. If `None`, the same `x_columns` names used to train the model are used.

        > **y_columns:** Columns with true output values to compare predicted values with. If `None`, the same
        `y_columns` names used to train the model are used.

        > **batch_size:** Batch size to use. If not specified, the entire dataset is used.

        **Returns**
        > Evaluation of the prediction using the `loss_function` provided in `df.nn.init(...)`.

        Examples
        ```python
        >>> df_test = pd.DataFrame({"x": [-1, 0, 1], "y": [0, 0, 1]})
        >>> df_test.model = df_train.nn.get_model()
        >>> df_test.nn.evaluate()
        ```
        """
        if x_columns is None:
            x_columns = self._df.model._x_columns
        if y_columns is None:
            y_columns = self._df.model._y_columns
        batch_size = batch_size if batch_size is not None else self._num_rows
        num_batches = _get_num_batches(num_rows=self._num_rows, batch_size=batch_size)

        losses = []
        for batch_number in range(num_batches):
            batch = _get_batch(self._df, batch_number, batch_size, x_columns, y_columns)
            losses.append(self._df.model.evaluate(batch.x, batch.y))

        return jnp.average(losses)
