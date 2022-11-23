from src.utils.parsers.cv_parser import parse_cv_result
from src.orchestrator.base_runner import BaseRunner

from src.utils.utils import build_model_pipeline
from src.orchestrator.config import GRIDSEARCH_KWARGS, RunConfig

from src.pipeline.taskloader import TaskLoader, TaskFrame
from src.pipeline.dataloader import DataLoader

from sklearn.model_selection import RepeatedKFold, GridSearchCV

import pandas as pd
import os

class Runner(BaseRunner):
    def __init__(
        self,
        dl: DataLoader,
        tl: TaskLoader,
    ) -> None:
        super().__init__(dl, tl)

        self.current_run_config: RunConfig = None
        self.cv = RepeatedKFold(n_repeats=3, n_splits=5, random_state=0)
    
    def prepare_run(self, 
        run_config,
    ):
        self.current_run_config: RunConfig = run_config
        return self
    
    def _check_if_file_exists(self, tf):
        return os.path.exists(f'{self._get_experiment_path()}/{tf.title}_{tf.frame_name}.csv')
    
    def _do_grid_search(self, tf, strategy, split_kwargs):
        model = build_model_pipeline(tf)
        return self.trainer.do_grid_search(
            tf,
            strategy,
            model, 
            self.current_run_config.grid_search_params,
            self.cv,
            split_kwargs=split_kwargs,
            search_kwargs=GRIDSEARCH_KWARGS,
        )
    
    def _do_retrain(self, tf, strategy, result_df, split_kwargs):
        model = build_model_pipeline(tf)
        model = parse_cv_result(model, result_df)
        model = self.trainer.do_retrain_model(tf, strategy, model, split_kwargs=split_kwargs)

        return model
    
    def _write_cv_result(self, 
        cv_result: GridSearchCV,
        tf: TaskFrame,
    ):
        if not os.path.exists(f'{self._get_experiment_path()}'):
            os.makedirs(f'{self._get_experiment_path()}')

        pd.DataFrame(cv_result.cv_results_) \
            .to_csv(f'{self._get_experiment_path()}/{tf.title}_{tf.frame_name}.csv')