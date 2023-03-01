from typing import Any, Dict, List

from src.pipeline.taskloader import TaskLoader, TaskFrame, TaskLoaderConfig
from src.settings.strategy import Strategy
from src.settings.tier import Tier

from src.parsers.cv_parser import to_cv_params, parse_cv_result
from src.utils.utils import get_project_root

import pandas as pd

from scipy.stats import pearsonr

from sklearn.compose import make_column_transformer, ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

class Trainer:
    def do_train_test_split(
        self,
        tf: TaskFrame,
        strategy: Strategy,
        test_size=0.3,
        shuffle=False,
        stratify=None,
    ):
        

        X_train = pd.read_csv(f'{get_project_root()}/data/preprocessed/x_train_{tf.tier}_{tf.frame_name}_{strategy}.csv')
        X_test = pd.read_csv(f'{get_project_root()}/data/preprocessed/x_test_{tf.tier}_{tf.frame_name}_{strategy}.csv')
        y_train = pd.read_csv(f'{get_project_root()}/data/preprocessed/y_train_{tf.tier}_{tf.frame_name}_{strategy}.csv')['metabolite_concentration'].ravel()
        y_test = pd.read_csv(f'{get_project_root()}/data/preprocessed/y_test_{tf.tier}_{tf.frame_name}_{strategy}.csv')['metabolite_concentration'].ravel()
        return X_train, X_test, y_train, y_test

        df = tf.x.reset_index()

        if strategy == strategy.ONE_VS_ALL:
            metabolite_id = tf.frame_name
            X_train, X_test = df[df['metabolite_id'] != metabolite_id], df[df['metabolite_id'] == metabolite_id]
            y_train, y_test = tf.y[X_train.index], tf.y[X_test.index]
            return X_train, X_test, y_train, y_test

        return train_test_split(
            df,
            tf.y,
            test_size=test_size,
            random_state=0,
            shuffle=shuffle,
            stratify=df[stratify] if stratify is not None else stratify
        )

    def do_grid_search(
        self,
        tf: TaskFrame, 
        model: TransformedTargetRegressor, 
        params: List[Dict], 
        cv, 
        split_kwargs: Dict,
        search_kwargs: Dict,
    ):
        print(f'Training: {tf.title}_{tf.frame_name}.csv')
        print(f"{tf.x.shape=} {tf.y.shape}")
        print(search_kwargs)

        X_train, _, y_train, _ = self.do_train_test_split(tf, **split_kwargs)

        search = GridSearchCV(
            model,
            to_cv_params(params),
            cv=cv,
            **search_kwargs,
        )
        search.fit(X_train, y_train)
        return search

    def do_retrain_model(
        self,
        tf: TaskFrame, 
        model: TransformedTargetRegressor, 
        split_kwargs: Dict
    ):
        X_train, X_test, y_train, y_test = self.do_train_test_split(tf, **split_kwargs)
        model.fit(X_train, y_train)
        return model