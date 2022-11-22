from dataclasses import dataclass
from typing import Dict

from src.settings.strategy import Strategy
from src.settings.tier import Tier

from src.pipeline.dataloader import DataLoaderConfig

GRIDSEARCH_KWARGS = dict(
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    refit=True,
    verbose=1,
    error_score='raise',
)

PATH_PREFIX = './data/results'

@dataclass
class RunConfig:
    experiment_id: str
    tier: Tier
    strategy: Strategy
    grid_search_params: Dict
    forced_training: bool = False
    forced_testing: bool = False

@dataclass
class ExplanationConfig:
    experiment_id: str