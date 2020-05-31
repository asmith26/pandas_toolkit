from typing import Callable

import haiku as hk
import jax.numpy as jnp
from sklearn.metrics import mean_squared_error

SKLEARN_METRICS = {"mean_squared_error": mean_squared_error}
SUPPORTED_LOSSES = list(SKLEARN_METRICS.keys())


class LossNotCurrentlySupportedException(Exception):
    def __init__(self, loss):
        super().__init__(f'Loss={loss} is not currently supported. Currently supported losses are: {SUPPORTED_LOSSES}')


def get_loss_function(net_transform: hk.Transformed, loss: str) -> Callable[[hk.Params, jnp.ndarray, jnp.ndarray], jnp.ndarray]:

    if loss in list(SKLEARN_METRICS.keys()):
        sklearn_metric_function = SKLEARN_METRICS[loss]

        def sklearn_metric_as_loss(params: hk.Params, x: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
            y_pred: jnp.ndarray = net_transform.apply(params, x)
            loss_value: jnp.ndarray = sklearn_metric_function(y_true, y_pred)  # sample_weight or **sklearn_metric_kwargs)
            return loss_value

        return sklearn_metric_as_loss
    else:
        raise LossNotCurrentlySupportedException(loss)
