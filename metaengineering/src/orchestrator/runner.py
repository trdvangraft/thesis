from src.utils.parsers.cv_parser import _fmt_regressor, get_architectures, parse_cv_result
from src.settings.strategy import Strategy
from src.orchestrator.trainer import Trainer

from src.utils.utils import TestResultStore, build_model_pipeline, get_generator, get_project_root
from src.orchestrator.config import GRIDSEARCH_KWARGS, PATH_PREFIX, RunConfig

from src.pipeline.taskloader import TaskLoader, TaskFrame
from src.pipeline.dataloader import DataLoader

from sklearn.model_selection import RepeatedKFold, GridSearchCV

import pandas as pd
import os
import glob
import pickle

class Runner:
    def __init__(
        self,
        dl: DataLoader,
        tl: TaskLoader,
    ) -> None:
        self.data_loader = dl
        self.task_loader = tl

        self.trainer = Trainer()
        self.current_run_config: RunConfig = None
        self.cv = RepeatedKFold(n_repeats=3, n_splits=5, random_state=0)
    
    def prepare_run(self, 
        run_config,
    ):
        self.current_run_config: RunConfig = run_config
    
    def run_training(
        self
    ): 
        if self.current_run_config.strategy == Strategy.ALL:
            self.run_training_strategy_all()
        elif self.current_run_config.strategy == Strategy.METABOLITE_CENTRIC:
            self.run_training_strategy_metabolite()
        elif self.current_run_config.strategy == Strategy.ONE_VS_ALL:
            self.run_training_strategy_one_vs_all()
        else:
            raise NotImplemented(f"{self.run_config.strategy} has not been implemenented")

    def run_training_strategy_all(self):
        split_kwargs = dict(
            stratify='metabolite_id',
            shuffle=True
        )
        self._run_training(split_kwargs)
    
    def run_training_strategy_metabolite(self):
        split_kwargs = dict(shuffle=False, stratify=None)
        self._run_training(split_kwargs)
    
    def run_training_strategy_one_vs_all(self):
        split_kwargs = dict(shuffle=False, stratify=None)
        self._run_training(split_kwargs)
    
    def _run_training(self, split_kwargs):
        gen = self._get_new_generator()
        for tf in gen:
            # print(tf)
            if self._check_if_file_exists(tf) and not self.current_run_config.forced_training:
                print(f"Result for {tf.title}_{tf.frame_name} already exists")
                continue
            cv_result = self._do_grid_search(tf, split_kwargs)
            self._write_cv_result(cv_result, tf)
    
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
        print(results_df)
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
    
    def _check_if_file_exists(self, tf):
        return os.path.exists(f'{self._get_experiment_path()}/{tf.title}_{tf.frame_name}.csv')
    
    def _do_grid_search(self, tf, split_kwargs):
        model = build_model_pipeline(tf)
        return self.trainer.do_grid_search(
            tf,
            model, 
            self.current_run_config.grid_search_params,
            self.cv,
            split_kwargs=split_kwargs,
            search_kwargs=GRIDSEARCH_KWARGS,
        )
    
    def _do_retrain(self, tf, result_df, split_kwargs):
        model = build_model_pipeline(tf)
        model = parse_cv_result(model, result_df)
        model = self.trainer.do_retrain_model(tf, model, split_kwargs=split_kwargs)

        return model
    
    def _get_new_generator(self):
        return get_generator(
            self.data_loader,
            self.task_loader,
            self.current_run_config.strategy,
            self.current_run_config.tier,
        )
    
    def _get_experiment_path(
        self
    ):
        return f"{get_project_root()}/data/results/{self.current_run_config.experiment_id}"
    
    def _get_model_path(
        self
    ):
        return f"{get_project_root()}/model/{self.current_run_config.experiment_id}"
    
    def _write_cv_result(self, 
        cv_result: GridSearchCV,
        tf: TaskFrame,
    ):
        if not os.path.exists(f'{self._get_experiment_path()}'):
            os.makedirs(f'{self._get_experiment_path()}')

        pd.DataFrame(cv_result.cv_results_) \
            .to_csv(f'{self._get_experiment_path()}/{tf.title}_{tf.frame_name}.csv')

    def _run_testing(self, results_df, split_kwargs):
        architectures = get_architectures(results_df)
        testResultStore = TestResultStore(self._get_experiment_path(), self.current_run_config.strategy)

        if not os.path.exists(f'{self._get_model_path()}'):
            print(f"{self._get_model_path()=}")
            os.makedirs(f'{self._get_model_path()}')

        if testResultStore.check_if_result_exists() and not self.current_run_config.forced_testing:
            return

        for architecture in architectures:
            print(architecture)

            for tf in self._get_new_generator():
                _result_df = results_df.copy() if architecture == 'all' else results_df[results_df['param_regressor__regressor'] == architecture].copy()

                if self.current_run_config.strategy != Strategy.ALL:
                    _result_df[_result_df['metabolite_id'] == tf.frame_name]

                _result_df = _result_df.sort_values('rank_test_score').iloc[[0]]

                model = self._do_retrain(tf, _result_df, split_kwargs)
                _, X_test, _, y_test = self.trainer.do_train_test_split(tf, **split_kwargs)

                if self.current_run_config.strategy == Strategy.ALL:
                    testResultStore.update_results(
                        'all', model, architecture,
                        X_test, y_test
                    )

                for metabolite_id in X_test['metabolite_id'].unique():
                    testResultStore.update_results(
                        metabolite_id, 
                        model,
                        architecture,
                        X_test[X_test['metabolite_id'] == metabolite_id], 
                        y_test.xs(metabolite_id, level='metabolite_id')
                    )
                
                with open(f'{self._get_model_path()}/{tf.title}_{tf.frame_name}.pickle', 'wb') as handle:
                    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        testResultStore.to_file()