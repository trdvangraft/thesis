from src.orchestrator.runner import Runner
from src.parsers.cv_parser import _fmt_regressor, get_architectures, parse_cv_result
from src.settings.strategy import Strategy

from src.orchestrator.config import GRIDSEARCH_KWARGS, PATH_PREFIX, RunConfig

from src.pipeline.taskloader import TaskLoader, TaskFrame
from src.pipeline.dataloader import DataLoader

from sklearn.model_selection import RepeatedKFold, GridSearchCV

class TrainRunner(Runner):
    def __init__(
        self,
        dl: DataLoader,
        tl: TaskLoader,
    ) -> None:
        super().__init__(dl, tl)
        self.cv = RepeatedKFold(n_repeats=3, n_splits=5, random_state=0)
    
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
            cv_result = self._do_grid_search(tf, self.current_run_config.strategy, split_kwargs)
            self._write_cv_result(cv_result, tf)