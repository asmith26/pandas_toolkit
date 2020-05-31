import unittest
from unittest.mock import MagicMock

from pandas_toolkit.nn.loss import LossNotCurrentlySupportedException, get_loss_function


class TestGetLossFunction(unittest.TestCase):
    def test_unsupported_loss_raises_error(self):
        loss = "some_unsupported_loss"
        with self.assertRaises(LossNotCurrentlySupportedException) as _:
            get_loss_function(MagicMock(), loss)
