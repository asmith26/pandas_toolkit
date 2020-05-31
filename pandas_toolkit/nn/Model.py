from typing import Callable, Any

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import optix
from jax.experimental.optix import InitUpdate

from pandas_toolkit.nn.loss import get_loss_function
from pandas_toolkit.utils.custom_types import Batch

OptState = Any


class Model(object):
    def __init__(self, net_function: Callable[[jnp.ndarray], jnp.ndarray], loss: str, optimizer: InitUpdate, batch: Batch):
        self.net_transform = hk.transform(net_function)
        self.optimizer = optimizer

        self.loss_function = get_loss_function(self.net_transform, loss)

        rng = jax.random.PRNGKey(42)
        self.params: hk.Params = self.net_transform.init(rng, batch.x)
        self.opt_state: OptState = optimizer.init(self.params)

    def evaluate(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return self.loss_function(self.params, x, y)

    # def hvplot_val_loss():

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.net_transform.apply(self.params, x)

    # @jax.jit
    def _update(self, x: jnp.ndarray, y: jnp.ndarray):
        """Learning rule (stochastic gradient descent)."""
        grads = jax.grad(self.loss_function)(self.params, x, y)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optix.apply_updates(self.params, updates)
