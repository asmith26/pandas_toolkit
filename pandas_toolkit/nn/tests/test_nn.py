import unittest

import haiku as hk
import jax.numpy as jnp
import pandas as pd
from jax.nn import relu

import pandas_toolkit.nn
from pandas_toolkit.nn.Model import Model


class TestInit(unittest.TestCase):
    def test_simple_relu_net(self):
        df_train = pd.DataFrame({"x": [0.0, 1.0], "y": [0, 1]})

        def net_function(x: jnp.ndarray) -> jnp.ndarray:
            net = hk.Sequential([relu])
            predictions: jnp.ndarray = net(x)
            return predictions

        df_train = df_train.nn.init(x_columns=["x"], y_columns=["y"], net_function=net_function, loss="mean_squared_error")

        for _ in range(10):  # num_epochs
            # df = df.nn.augment()
            # df = df.nn.shuffle()
            df_train = df_train.nn.update()

        self.assertTrue(isinstance(df_train.model, Model), "df.model is not of type pandas_toolkit.nn.Model")

        actual_loss = df_train.nn.evaluate()
        expected_loss = 0
        self.assertEqual(expected_loss, actual_loss)

        df_test = pd.DataFrame({"x": [-1, 0, 1]})
        df_test.model = df_train.nn.get_model()

        actual_predictions = df_test.nn.predict()
        expected_predictions = jnp.array([0, 0, 1])
        self.assertTrue(
            (actual_predictions == expected_predictions).all(),
            f"expected_predictions={expected_predictions}!={actual_predictions}=actual_predictions",
        )
