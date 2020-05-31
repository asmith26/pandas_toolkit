from typing import Callable, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.experimental import optix
from jax.experimental.optix import InitUpdate

from pandas_toolkit.nn.Model import Model
from pandas_toolkit.utils.custom_types import Batch


def get_batch_numbers(num_rows: int, batch_size: Optional[int]) -> Tuple[int, jnp.ndarray]:
    batch_numbers: jnp.ndarray
    if batch_size is None:
        num_batches = 1
        batch_numbers = jnp.zeros(num_rows)
    else:
        num_batches = int(num_rows / batch_size) + 1
        batch_numbers = jnp.repeat(jnp.arange(num_batches), batch_size)[:num_rows]
    return num_batches, batch_numbers


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
        df = self._df.sample(frac=1)
        df.num_batches, df["batch_number"] = get_batch_numbers(num_rows=len(df), batch_size=batch_size)
        df.set_index("batch_number", inplace=True)
        self._df = df

        batch0 = get_batch(df, x_columns, y_columns, batch_number=0)
        self._df.model = Model(net_function, loss, optimizer, batch0)

        self._df._x_columns = x_columns
        self._df._y_columns = y_columns

        return self._df

    # def get_model(self) -> Model:
    #     return self._df.model.copy()

    def update(self) -> pd.DataFrame:
        for batch_number in range(self._df.num_batches):
            batch = get_batch(self._df, self._df._x_columns, self._df._y_columns, batch_number)
            self._df.model._update(batch.x, batch.y)

        return self._df
