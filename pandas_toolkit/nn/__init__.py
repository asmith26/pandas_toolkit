import math
from typing import Callable, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.experimental import optix
from jax.experimental.optix import InitUpdate

from pandas_toolkit.nn.Model import Model
from pandas_toolkit.utils.custom_types import Batch


def _get_num_batches(num_rows: int, batch_size: Optional[int]) -> Tuple[int, jnp.ndarray]:
    if batch_size is None:
        return 1
    num_batches = num_rows / batch_size
    if math.ceil(num_batches) == math.floor(num_batches):
        return num_batches
    return int(num_batches) + 1


def get_batch(df: pd.DataFrame, x_columns: List[str], y_columns: List[str], batch_number: int) -> Batch:
    df_batch = df.loc[batch_number]
    x_train: np.ndarray = df_batch[x_columns].values
    y_train: np.ndarray = df_batch[y_columns].values
    return Batch(x=jnp.array(x_train), y=jnp.array(y_train))


@pd.api.extensions.register_dataframe_accessor("nn")
class NeuralNetworkAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df = df

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
        self._df.num_batches
        self._df.model = Model(net_function, loss, optimizer, batch0)

        self._df.model._x_columns = x_columns
        self._df.model._y_columns = y_columns

        return self._df

    def get_model(self) -> Model:
        return self._df.model.copy()

    def hvplot_losses(self):
        from streamz import Stream
        from streamz.dataframe import DataFrame
        import hvplot.streamz

        self.sdf = DataFrame(Stream(), example=pd.DataFrame({"epoch": [], "train_loss": [], "validation_loss": []}))
        return self.sdf.hvplot.line(x="epoch", y=["train_loss", "validation_loss"])

    def update(self, df_validation: pd.DataFrame, hvplot_losses: bool = False) -> pd.DataFrame:
        # df = self._df.sample(frac=1)
        # df_train = df.query(f"{is_validation_data_column} == 0")
        # df_validation = df.query(f"{is_validation_data_column} == 1")
        #
        # df.num_batches, df.train["batch_number"] = get_batch_numbers(
        #     num_rows=len(df.train), batch_size=batch_size)
        # df.validation["batch_number"] = -1
        # self._df =
        # df.train.set_index("batch_number", inplace=True)
        #
        # batch0 = get_batch(df, x_columns, y_columns, batch_number=0)

        for batch_number in range(self._df.num_batches):
            batch = get_batch(self._df, self._df.model._x_columns, self._df.model._y_columns, batch_number)
            self._df.model._update(batch.x, batch.y)

        self._df.model.num_epochs += 1
        if hvplot_losses:
            df_validation.model = self.get_model()
            df_losses = pd.DataFrame({"epoch": [self._df.model.num_epochs],
                                      "train_loss": self.evaluate().tolist(),
                                      "validation_loss": df_validation.evaluate().tolist()})
            self.sdf.emit(df_losses)
        return self._df

    def predict(self, x_columns: List[str] = None) -> pd.Series:
        if x_columns is None:
            x_columns = self._df.model._x_columns
        x = jnp.array(self._df[x_columns].values)
        predictions: jnp.array =  self._df.model.predict(x)
        num_features = predictions.shape[1]
        if num_features == 1:
            return predictions.flatten()
        return predictions

    def evaluate(self, x_columns: List[str] = None, y_columns: List[str] = None) -> pd.Series:
        if x_columns is None:
            x_columns = self._df.model._x_columns
        if y_columns is None:
            y_columns = self._df.model._y_columns
        x = jnp.array(self._df[x_columns].values)
        y = jnp.array(self._df[y_columns].values)
        return self._df.model.evaluate(x, y)
