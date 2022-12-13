from src.orchestrator.explainer import Explainer
from src.orchestrator.test_runner import TestRunner
from src.orchestrator.train_runner import TrainRunner
from src.settings.strategy import Strategy

from src.pipeline.dataloader import DataLoader
from src.pipeline.taskloader import TaskLoader
from src.pipeline.config import TaskLoaderConfig, DataLoaderConfig

from src.orchestrator.runner import Runner
from src.orchestrator.config import RunConfig, ExplanationConfig

class BaseOrchestrator:
    def __init__(
        self,
    ) -> None:
        self.dl_config = None
        self.tl_config = None
    
    def prepare_orchestrator(
        self, 
        data_loader_config: DataLoaderConfig,
        task_loader_config: TaskLoaderConfig,
    ):
        self.dl_config = data_loader_config
        self.tl_config = task_loader_config
    
    def get_dataloader(self):
        dl = DataLoader()
        dl.prepare_dataloader(self.dl_config)
        return dl
    
    def get_taskloader(self):
        tl = TaskLoader()
        tl.prepare_taskloader(self.tl_config)
        return tl
    
    def run(self):
        pass
    
    def _is_valid_state(self):
        return self.dl_config is not None and self.tl_config is not None

class SklearnOrchestrator(BaseOrchestrator):
    def __init__(self) -> None:
        self.run_config = None
        self.explain_config = None
        self.visualizer_config = None

    def prepare_orchestrator(self, 
        data_loader_config: DataLoaderConfig,
        task_loader_config: TaskLoaderConfig,
        run_config: RunConfig,
        explain_config: ExplanationConfig,
    ):
        super().prepare_orchestrator(data_loader_config, task_loader_config)
        self.run_config = run_config
        self.explain_config = explain_config

    def run(self):
        if not self._is_valid_state():
            raise ValueError("Missing config file")
        
        dl = self.get_dataloader()
        tl = self.get_taskloader()
        
        TrainRunner(dl, tl).prepare_run(self.run_config).run_training()
        TestRunner(dl, tl).prepare_run(self.run_config).run_testing()

        explainer = Explainer(dl, tl)
        explainer.prepare_explainer(self.explain_config)
        explainer.run()

    def _is_valid_state(self):
        return super()._is_valid_state() and self.run_config is not None and self.explain_config is not None

class SGDOrchestrator(BaseOrchestrator):
    def __init__(self) -> None:
        super().__init__()
    
    def prepare_orchestrator(self, 
        data_loader_config: DataLoaderConfig, 
        task_loader_config: TaskLoaderConfig
    ):
        super().prepare_orchestrator(data_loader_config, task_loader_config)
    
    def run(self):
        if not self._is_valid_state():
            raise ValueError("Missing config file")
        
        dl = self.get_dataloader()
        tl = self.get_taskloader()

        

    
    def _is_valid_state(self):
        return super()._is_valid_state()
    
