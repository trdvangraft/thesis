from src.orchestrator.runner import Runner
from src.settings.strategy import Strategy

from src.parsers.cv_parser import _fmt_regressor, get_architectures, parse_cv_result
from src.utils.test_result_store import TestResultStore

from src.pipeline.taskloader import TaskLoader
from src.pipeline.dataloader import DataLoader

import pandas as pd
import numpy as np
import os
import glob
import pickle

from sklearn.metrics import mean_absolute_error
from joblib import Parallel, delayed

class TestRunner(Runner):
    def __init__(self, dl: DataLoader, tl: TaskLoader) -> None:
        super().__init__(dl, tl)

    def run_testing(self):
        if self.current_run_config.strategy == Strategy.ALL:
            self.run_testing_strategy_all()
        elif self.current_run_config.strategy == Strategy.METABOLITE_CENTRIC:
            self.run_testing_strategy_metabolite()
        elif self.current_run_config.strategy == Strategy.ONE_VS_ALL:
            self.run_testing_strategy_one_vs_all()
        else:
            raise NotImplemented(f"{self.current_run_config.strategy} has not been implemenented")

    def run_testing_strategy_all(self):
        results_df = pd.read_csv(f'{self._get_experiment_path()}/{self.current_run_config.strategy}_all.csv')
        results_df = _fmt_regressor(results_df)
        split_kwargs = dict(
            stratify='metabolite_id',
            shuffle=True
        )

        self._run_testing(results_df, split_kwargs)
    
    def run_testing_strategy_metabolite(self):
        paths = glob.glob(self._get_experiment_path() + "/*METABOLITE_CENTRIC*.csv")
        results_df = self.result_df(paths)
        split_kwargs = dict(shuffle=False, stratify=None)
        self._run_testing(results_df, split_kwargs)
 
    def run_testing_strategy_one_vs_all(self):
        paths = glob.glob(self._get_experiment_path() + "/*ONE_VS_ALL*.csv")
        results_df = self.result_df(paths)
        split_kwargs = dict(shuffle=False, stratify=None)

        self._run_testing(results_df, split_kwargs)

    def result_df(self, paths):
        metabolites_names = [path.rsplit('/', 1)[1].removesuffix('.csv').removeprefix(f'{self.current_run_config.strategy}_') for path in paths]
        results_df = pd.concat([
            pd.read_csv(path).assign(metabolite_id=metabolite_name) 
            for path, metabolite_name in zip(paths, metabolites_names)
        ])
        results_df = _fmt_regressor(results_df)
        return results_df
    
    def _run_testing(self, results_df, split_kwargs):
        architectures = get_architectures(results_df)
        testResultStore = TestResultStore(
            self._get_experiment_path(), 
            self.current_run_config.strategy,
            'sklearn'
        )

        if not os.path.exists(f'{self._get_model_path()}'):
            print(f"{self._get_model_path()=}")
            os.makedirs(f'{self._get_model_path()}')

        if testResultStore.check_if_result_exists() and not self.current_run_config.forced_testing:
            return

        print(f"{architectures}")
        for architecture in architectures:
            print(f"{architecture=}")

            for tf in self._get_new_generator():
                _result_df = results_df.copy() if architecture == 'all' else results_df[results_df['param_regressor'] == architecture].copy()

                if self.current_run_config.strategy != Strategy.ALL:
                    _result_df[_result_df['metabolite_id'] == tf.frame_name]

                _result_df = _result_df.sort_values('rank_test_score').iloc[[0]]

                # models = [
                #     self.retrain_architecture(tf, split_kwargs, _result_df)
                #     for run_id in range(10)
                # ]

                models = Parallel(n_jobs=-1)(delayed(self.retrain_architecture)(tf, split_kwargs, _result_df) for run_id in range(10))

                _, X_test, _, y_test = self.trainer.do_train_test_split(tf, self.current_run_config.strategy, **split_kwargs)
                for run_id in range(10):
                    if self.current_run_config.strategy == Strategy.ALL:
                        for metabolite_id in X_test['metabolite_id'].unique():
                            testResultStore.update_results(
                                f"{run_id}_{metabolite_id}",
                                models[run_id].predict, 
                                architecture,
                                X_test[X_test['metabolite_id'] == metabolite_id], 
                                y_test.xs(metabolite_id, level='metabolite_id')
                            )
                    else:
                        metabolite_id = tf.frame_name
                        testResultStore.update_results(
                            f"{run_id}_{metabolite_id}",
                            models[run_id].predict,
                            architecture,
                            X_test, 
                            y_test
                        )

                if architecture == 'all':
                    model_perfomances = np.array([
                        mean_absolute_error(
                            testResultStore.pred_results[f"{run_id}_{metabolite_id}_all"]['y_true'],
                            testResultStore.pred_results[f"{run_id}_{metabolite_id}_all"]['y_pred'],
                        )
                        for run_id in range(10)
                    ])

                    idx = np.argmin(model_perfomances)


                    with open(f'{self._get_model_path()}/{tf.title}_{tf.frame_name}.pickle', 'wb') as handle:
                        pickle.dump(models[idx], handle, protocol=pickle.HIGHEST_PROTOCOL)

        testResultStore.to_file()

    def retrain_architecture(self, tf, split_kwargs, _result_df):
        model = self._do_retrain(tf, self.current_run_config.strategy, _result_df, split_kwargs)
        return model

    def get_metabolite_id(self, tf):
        if self.current_run_config.strategy == Strategy.ALL:
            return 'all'
        return tf.frame_name