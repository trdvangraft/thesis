from cmath import isnan
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.pipeline.dataloader import DataLoaderConfig, DataLoader
from src.pipeline.taskloader import TaskLoaderConfig, TaskLoader, TaskFrame

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


def gather_results(paths: List[str]):
    metabolites_names = [path.rsplit('/', 1)[1].removesuffix('.csv').removeprefix('Strategy.METABOLITE_CENTRIC_') for path in paths]
    df = pd.concat([
        pd.read_csv(path).assign(metabolite_id=metabolite_name)
        for path, metabolite_name in zip(paths, metabolites_names)
    ])
    return df

def get_generator(
    dl: DataLoader,
    tl: TaskLoader,
    strategy:Strategy=Strategy.METABOLITE_CENTRIC,
    tier=Tier.TIER1,
    data_config: DataLoaderConfig = DataLoaderConfig(),
    task_config: TaskLoaderConfig = TaskLoaderConfig()
):
    df = dl.get_dataframe(data_config)
    gen = tl.prepare_task(df, tier, task_config).build(strategy)
    return gen

def build_model_pipeline(
    tf: TaskFrame = None,
):
    numeric_features = ['enzyme_concentration'] + tf.x.columns.to_list()
    numeric_features = list(filter(lambda x: x in tf.x.columns.to_list(), numeric_features))
    numeric_transformer = Pipeline(
        steps=[
            ('scaler', StandardScaler())
        ]
    )

    prepocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            # ('cat', categorical_transformer, cat_features),
        ],
        remainder='drop'
    )

    estimator = DecisionTreeRegressor()

    clf = Pipeline(
        steps=[
            ('preprocessor', prepocessor),
            # ('pca', PCA()),
            ('regressor', estimator),
        ]
    )

    model = TransformedTargetRegressor(
        regressor=clf,
        transformer=None,
    )
    return model

class TestResultStore:
    def __init__(
        self,
        experiment_path: str,
        strategy: str,
    ) -> None:
        self.results = defaultdict(lambda: list())
        self.pred_results = defaultdict(lambda: dict())

        self.experiment_path = experiment_path
        self.strategy = strategy

    def update_results(
        self,
        key: str,
        model: TransformedTargetRegressor,
        architecture: str,
        X_test: np.array,
        y_test: np.array
    ):
        if len(X_test) < 2:
            return

        correlation = pearsonr(y_test, model.predict(X_test))[0]

        if np.isnan(correlation):
            correlation = 0

        self.results[f"{key}_{architecture}"].append(correlation)
        self.pred_results[f"{key}_{architecture}"].update({
            'y_true': y_test.values,
            'y_pred': model.predict(X_test),
            'architecture': architecture,
            'metabolite_id': key,
            'correlation': correlation,
        })

    def to_file(self):
        pd.DataFrame.from_dict(self.results).to_csv(f'{self.experiment_path}/best_model_performance_{self.strategy}.csv')
        pd.DataFrame.from_dict(self.pred_results).to_json(f'{self.experiment_path}/best_model_prediction_performance_{self.strategy}.json')
