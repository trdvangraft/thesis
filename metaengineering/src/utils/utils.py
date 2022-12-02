import os
from collections import defaultdict
from typing import List

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
    strategy: Strategy,
    tier: Tier
):
    df = dl.get_dataframe()
    gen = tl.prepare_task(df).build(strategy)
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

def build_config(
    strategy: Strategy,
    tier: Tier,
    params,
    additional_frames=[],
    additional_filters=[],
    additional_transforms=[],
    forced_training=False,
    forced_testing=False, 
    data_throttle=1,
):
    dl_config = DataLoaderConfig(
        additional_frames=additional_frames,
        additional_filters=additional_filters,
        additional_transforms=additional_transforms,
    )

    tl_config = TaskLoaderConfig(
        data_throttle=data_throttle,
        tier=tier
    )

    run_config = RunConfig(
        experiment_id=str(tier),
        tier=tier,
        strategy=strategy,
        grid_search_params=params,
        forced_training=forced_training,
        forced_testing=forced_testing,
    )

    exp_config = ExplanationConfig(
        experiment_id=str(tier),
        tier=tier,
        strategy=strategy
    )

    return dl_config, tl_config, run_config, exp_config

def get_project_root():
    return Path(__file__).parent.parent.parent

def make_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)