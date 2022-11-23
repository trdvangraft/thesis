from src.orchestrator.trainer import Trainer

from src.utils.utils import get_generator, get_project_root
from src.orchestrator.config import  RunConfig

from src.pipeline.taskloader import TaskLoader
from src.pipeline.dataloader import DataLoader

class BaseRunner:
    def __init__(
        self,
        dl: DataLoader,
        tl: TaskLoader,
    ) -> None:
        self.data_loader = dl
        self.task_loader = tl

        self.trainer = Trainer()
        self.current_run_config: RunConfig = None
    
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
    
    def _get_explanation_path(
        self
    ):
        return f"{get_project_root()}/explanation/{self.current_run_config.experiment_id}"