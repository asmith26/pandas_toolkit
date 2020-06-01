import unittest

import haiku as hk
import jax.numpy as jnp
import pandas as pd
from jax.nn import relu

import pandas_toolkit.nn
from pandas_toolkit.nn.Model import Model


class TestInit(unittest.TestCase):
    def test_simple_relu_net(self):
        df = pd.DataFrame({"x": [0.0, 1.0], "y": [0, 1]}, index=[0, 1])

        def net_function(x: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([relu])
            predictions: jnp.ndarray = net(x)
            return predictions

        df = df.nn.init(x_columns=["x"], y_columns=["y"], net_function=net_function, loss="mean_squared_error")

        for _ in range(10):  # num_epochs
            # df = df.nn.augment()
            # df = df.nn.shuffle()
            df = df.nn.update()

        self.assertTrue(isinstance(df.model, Model), "df.model is not of type pandas_toolkit.nn.Model")

        actual_predictions = df.model.predict(x=jnp.array([[-1], [0], [1]]))
        expected_predictions = jnp.array([[0], [0], [1]])
        self.assertTrue(
            (actual_predictions == expected_predictions).all(),
            f"expected_predictions={expected_predictions}!={actual_predictions}=actual_predictions",
        )

        actual_loss = df.model.evaluate(x=jnp.array([[-1], [0], [1]]), y=jnp.array([[0], [0], [1]]))
        expected_loss = 0
        self.assertEqual(expected_loss, actual_loss)
