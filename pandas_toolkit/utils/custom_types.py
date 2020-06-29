from collections import namedtuple
from typing import TypeVar

Batch = namedtuple("Batch", "x y")
ModelType = TypeVar("ModelType", bound="Model")
