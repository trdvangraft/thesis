from src.settings.strategy import Strategy

from src.pipeline.dataloader import DataLoader
from src.pipeline.taskloader import TaskLoader
from src.pipeline.config import TaskLoaderConfig, DataLoaderConfig

from src.orchestrator.runner import Runner
from src.orchestrator.config import RunConfig, ExplanationConfig

class Orchestrator:
    def __init__(self) -> None:
        self.dl_config = None
        self.tl_config = None
        self.run_config = None
        self.explain_config = None
        self.visualizer_config = None

    def prepare_orchestrator(self, 
        data_loader_config: DataLoaderConfig,
        task_loader_config: TaskLoaderConfig,
        run_config: RunConfig,
        explain_config: ExplanationConfig,
    ):
        self.dl_config = data_loader_config
        self.tl_config = task_loader_config
        self.run_config = run_config
        self.explain_config = explain_config

    def run(self):
        if not self._is_valid_state():
            raise ValueError("Missing config file")
        
        dl = DataLoader()
        dl.prepare_dataloader(self.dl_config)

        tl = TaskLoader()
        tl.prepare_taskloader(self.tl_config)

        runner = Runner(dl, tl)
        runner.prepare_run(self.run_config)
        runner.run_training()
        runner.run_testing()


        

    def _is_valid_state(self):
        return self.dl_config and self.tl_config and self.run_config and self.explain_config