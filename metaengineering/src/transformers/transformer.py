from typing import Any, Callable
from src.settings import DataOrientation

import pandas as pd


class Transformer:
    def __call__(self, df: pd.DataFrame) -> Any:
        pass


class ScaleTransformer(Transformer):
    def __init__(self, scaler: Callable) -> None:
        super().__init__()
        self.scaler = scaler

    def __call__(self, df: pd.DataFrame) -> Any:
        return self.scaler(df)
