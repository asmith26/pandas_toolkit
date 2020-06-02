import math
from typing import Callable, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
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


def _get_batch(df_train: pd.DataFrame, batch_number: int, batch_size: int, x_columns: List[str], y_columns: List[str]) -> Batch:
    start_batch_idx = batch_number * batch_size
    end_batch_idx = (batch_number + 1) * batch_size
    df_train_batch = df_train.iloc[start_batch_idx:end_batch_idx]
    return Batch(x=jnp.array(df_train_batch.loc[:, x_columns].values), y=jnp.array(df_train_batch.loc[:, y_columns].values))


@pd.api.extensions.register_dataframe_accessor("nn")
class NeuralNetworkAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df_train = df

    def init(
        self,
        x_columns: List[str],
        y_columns: List[str],
        net_function: Callable[[jnp.ndarray], jnp.ndarray],
        loss: str,
        optimizer: InitUpdate = optix.adam(learning_rate=1e-3),
        # num_epochs: int=1, DON'T DO THIS, JUST USE A FOR LOOP
        batch_size: int = None,
        # plot_val_loss=False
    ) -> pd.DataFrame:
        num_rows = len(self._df_train)
        self._df_train._num_batches = _get_num_batches(num_rows=num_rows, batch_size=batch_size)
        self._df_train._batch_size = batch_size if batch_size is not None else num_rows

        num_features = len(x_columns)
        example_x = jnp.zeros(shape=[self._df_train._batch_size, num_features])
        self._df_train.model = Model(net_function, loss, optimizer, example_x)

        self._df_train.model._x_columns = x_columns
        self._df_train.model._y_columns = y_columns

        return self._df_train

    def get_model(self) -> Model:
        return self._df_train.model.copy()

    def hvplot_losses(self):  # pragma: no cover
        from streamz import Stream
        from streamz.dataframe import DataFrame
        import hvplot.streamz

        self.sdf = DataFrame(Stream(), example=pd.DataFrame({"epoch": [], "train_loss": [], "validation_loss": []}))
        return self.sdf.hvplot.line(x="epoch", y=["train_loss", "validation_loss"])

    def update(self, df_validation: pd.DataFrame, hvplot_losses: bool = False) -> pd.DataFrame:
        df_train = self._df_train.sample(frac=1)

        for batch_number in range(self._df_train._num_batches):
            batch = _get_batch(df_train, batch_number, self._df_train._batch_size, self._df_train.model._x_columns, self._df_train.model._y_columns)
            self._df_train.model._update(batch.x, batch.y)

        self._df_train.model.num_epochs += 1
        if hvplot_losses:  # pragma: no cover
            df_validation.model = self.get_model()
            df_losses = pd.DataFrame({"epoch": [self._df_train.model.num_epochs],
                                      "train_loss": self.evaluate().tolist(),
                                      "validation_loss": df_validation.nn.evaluate().tolist()})
            self.sdf.emit(df_losses)
        return self._df_train

    def predict(self, x_columns: List[str] = None) -> pd.Series:
        if x_columns is None:
            x_columns = self._df_train.model._x_columns
        x = jnp.array(self._df_train[x_columns].values)
        predictions: jnp.array =  self._df_train.model.predict(x)
        num_features = predictions.shape[1]
        if num_features == 1:
            return predictions.flatten()
        return predictions

    def evaluate(self, x_columns: List[str] = None, y_columns: List[str] = None) -> pd.Series:
        if x_columns is None:
            x_columns = self._df_train.model._x_columns
        if y_columns is None:
            y_columns = self._df_train.model._y_columns
        x = jnp.array(self._df_train[x_columns].values)
        y = jnp.array(self._df_train[y_columns].values)
        return self._df_train.model.evaluate(x, y)
