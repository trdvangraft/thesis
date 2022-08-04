from dataclasses import dataclass, asdict

import pandas as pd
from src.models import BaseModel

import statsmodels.api as sm


@dataclass
class GLMConfig:
    familiy: sm.families.Family = sm.families.Gamma()
    offset = None
    exposure = None
    freq_weights = None
    var_weights = None
    missing = 'none'


class GLM(BaseModel):
    def __init__(self, config: GLMConfig = None) -> None:
        super().__init__()
        self.title = "Generalized linear model"
        self.model = None

        if config is None:
            config = GLMConfig()
        self.config = config

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        self.model = sm.GLM(y, x, **asdict(self.config))
        res = self.model.fit()
        return res
