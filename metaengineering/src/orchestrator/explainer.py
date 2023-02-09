from collections import defaultdict

from lime import lime_tabular

from shap import KernelExplainer, kmeans, Explanation

import pandas as pd
import numpy as np
import pickle
from src.orchestrator.config import ExplanationConfig, RunConfig
from src.orchestrator.base_runner import BaseRunner
from src.pipeline.dataloader import DataLoader
from src.pipeline.taskloader import TaskLoader, TaskFrame
from src.settings.strategy import Strategy

from src.utils.utils import get_generator, make_path_if_not_exists

import os

class Explainer(BaseRunner):
    def __init__(self, dl: DataLoader, tl: TaskLoader) -> None:
        super().__init__(dl, tl)

    def prepare_explainer(self, run_config: ExplanationConfig):
        self.current_run_config: ExplanationConfig = run_config
    
    def run(self):
        if self.current_run_config.strategy == Strategy.ALL:
            self.run_explanation_all()
        elif self.current_run_config.strategy == Strategy.METABOLITE_CENTRIC:
            self.run_explanation_metabolite()
        elif self.current_run_config.strategy == Strategy.ONE_VS_ALL:
            self.run_explanation_one_vs_all()
        else:
            raise NotImplemented(f"{self.run_config.strategy} has not been implemenented")
    
    def run_explanation_all(self):
        split_kwargs = dict(
            stratify='metabolite_id',
            shuffle=True
        )
        self._run_explanation(split_kwargs)

    def run_explanation_metabolite(self):
        split_kwargs = dict(shuffle=False, stratify=None)
        self._run_explanation(split_kwargs)

    def run_explanation_one_vs_all(self):
        split_kwargs = dict(shuffle=False, stratify=None)
        self._run_explanation(split_kwargs)
    
    def _run_explanation(self, split_kwargs):
        for tf in self._get_new_generator():
            X_train, X_test, y_train, y_test = self.trainer.do_train_test_split(tf, strategy=self.current_run_config.strategy, **split_kwargs)
            self.model = self.load_model(tf)
            self.lime_explanation(tf, X_train, X_test, y_train, y_test)
            self.shap_explanation(tf, X_train, X_test)

    def lime_explanation(
        self,
        tf: TaskFrame,
        X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.DataFrame, y_test: pd.DataFrame
    ):
        path = f"{self._get_explanation_path()}/lime"
        real_path = f"{path}/{self.current_run_config.strategy}_{tf.frame_name}.json"
        make_path_if_not_exists(path)

        if os.path.exists(real_path) and not self.current_run_config.forced_lime:
            return

        explanation_result = defaultdict(list)

        _X_train = X_train[X_train.columns.difference(['KO_ORF', 'metabolite_id', 'ORF'])]
        _X_test = X_test[X_test.columns.difference(['KO_ORF', 'metabolite_id', 'ORF'])]
        rm_columns = len(X_train.columns) - len(_X_train.columns)

        explainer = lime_tabular.LimeTabularExplainer(
            _X_train.values, 
            feature_names=_X_train.columns, 
            verbose=True, 
            mode='regression',
            discretize_continuous=False
        )

        for i in range(len(X_test)):
            exp = explainer.explain_instance(_X_test.iloc[i], self.get_lime_predict_fn(X_train.columns, rm_columns), num_features=100)
            explanation_result['KO_ORF'].append(X_test.iloc[i]['KO_ORF'])

            if self.current_run_config.strategy == Strategy.ALL:
                explanation_result['metabolite_id'].append(X_test.iloc[i]['metabolite_id'])
            else:
                explanation_result['metabolite_id'].append(tf.frame_name)
            explanation_result['exp_enzymes'].append([enzyme for enzyme, _ in exp.as_list()])
            explanation_result['exp_weights'].append([weight for _, weight in exp.as_list()])
            explanation_result['y_true'].append(exp.predicted_value)
            explanation_result['y_pred'].append(y_test.iloc[i])

        pd.DataFrame.from_dict(explanation_result).to_json(real_path)
    
    def get_lime_predict_fn(self, columns, rm_columns):
        model = self.model
        def lime_predict(x: np.array):
            _x = pd.DataFrame(
                np.append(np.zeros(shape=(x.shape[0], rm_columns)), x, axis=1),
                columns=columns
            )
            return model.predict(_x)
        return lime_predict

    def shap_explanation(
        self,
        tf: TaskFrame,
        X_train: pd.DataFrame, X_test: pd.DataFrame,
    ):
        X_train_summary = kmeans(self.apply_shap_transform(X_train), 12)
        ex = KernelExplainer(self.get_shap_predict_fn(), X_train_summary, seed=0)
        selected_feature_names = np.vectorize(lambda x: x.removeprefix("num__"))(self.model.regressor_[:-1].get_feature_names_out())

        metabolites_of_interests = ['pyr', 'pep', 'dhap', 'all']

        if self.current_run_config.strategy == Strategy.ALL:
            for id in metabolites_of_interests:
                self.get_shap_for_metabolite(X_test, ex, selected_feature_names, id)
        else:
            if tf.frame_name in metabolites_of_interests:
                self.get_shap_for_metabolite(X_test, ex, selected_feature_names, tf.frame_name)
    
    def get_shap_for_metabolite(
        self,
        X_test: pd.DataFrame,
        ex: KernelExplainer,
        feature_names,
        metabolite_id: str = 'all',
    ):
        path = f"{self._get_explanation_path()}/shap"
        make_path_if_not_exists(path)

        if os.path.exists(f"{path}/{self.current_run_config.strategy}_{metabolite_id}.pickle") and not self.current_run_config.forced_shap:
            return

        if self.current_run_config.strategy == Strategy.ALL:
            _df = X_test[X_test['metabolite_id'] == metabolite_id] if metabolite_id != 'all' else X_test
        else:
            _df = X_test

        num_instances = len(_df)
        sv = ex.shap_values(self.apply_shap_transform(_df), gc_collect=True)
        exp = Explanation(
            values=sv, 
            base_values=np.array([ex.expected_value] * num_instances),
            data=self.apply_shap_transform(_df), 
            feature_names=feature_names
        )

        with open(f"{path}/{self.current_run_config.strategy}_{metabolite_id}.pickle", 'wb') as f:
            pickle.dump(exp, f)

        return exp
    
    def load_model(self, tf):
        with open(f'{self._get_model_path()}/{tf.title}_{tf.frame_name}.pickle', 'rb') as handle:
            print(f'{self._get_model_path()}/{tf.title}_{tf.frame_name}.pickle')
            model = pickle.load(handle)
        return model
    
    def apply_shap_transform(self, X):
        _X = X
        for step in self.model.regressor_.steps[:-1]:
            _X = step[1].transform(_X)
        return _X
    
    def get_shap_predict_fn(self):
        model = self.model
        def shap_predict(X: pd.DataFrame):
            return model.regressor_.named_steps['regressor'].predict(X)
        return shap_predict

    