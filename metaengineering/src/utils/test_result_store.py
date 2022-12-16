import os
from collections import defaultdict
from typing import List, Callable

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

from src.pipeline.dataloader import DataLoader
from src.pipeline.taskloader import TaskLoader, TaskFrame
from src.pipeline.config import DataLoaderConfig, TaskLoaderConfig
from src.orchestrator.config import ExplanationConfig, RunConfig

from src.settings.strategy import Strategy
from src.settings.tier import Tier

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, GammaRegressor

class TestResultStore:
    def __init__(
        self,
        experiment_path: str,
        strategy: str,
        runner: str,
    ) -> None:
        self.results = defaultdict(lambda: list())
        self.pred_results = defaultdict(lambda: dict())
        self.runner = runner

        self.experiment_path = experiment_path
        self.strategy = strategy

    def update_results(
        self,
        key: str,
        predict_fn: Callable[[pd.DataFrame], np.array],
        # model: TransformedTargetRegressor,
        architecture: str,
        X_test: np.array,
        y_test: np.array
    ):
        if len(X_test) < 2:
            return

        correlation = pearsonr(y_test, predict_fn(X_test))[0]

        if np.isnan(correlation):
            correlation = 0

        self.results[f"{key}_{architecture}"].append(correlation)
        self.pred_results[f"{key}_{architecture}"].update({
            'y_true': y_test.values,
            'y_pred': predict_fn(X_test),
            'architecture': architecture,
            'metabolite_id': key,
            'correlation': correlation,
        })

    def to_file(self):
        pd.DataFrame.from_dict(self.results).to_csv(f'{self.experiment_path}/best_model_performance_{self.runner}_{self.strategy}.csv')
        pd.DataFrame.from_dict(self.pred_results).to_json(f'{self.experiment_path}/best_model_prediction_performance_{self.runner}_{self.strategy}.json')
    
    def check_if_result_exists(self):
        return os.path.exists(f'{self.experiment_path}/best_model_performance_{self.runner}_{self.strategy}.csv')