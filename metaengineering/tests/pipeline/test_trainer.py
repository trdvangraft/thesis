import unittest
from metaengineering.src.pipeline.taskloader import TaskLoaderConfig
from metaengineering.src.settings.strategy import Strategy
from metaengineering.src.settings.tier import Tier

from src.pipeline.dataloader import DataLoader
from src.pipeline.taskloader import TaskFrame, TaskLoader
from src.pipeline.trainer import Trainer

import pandas as pd
from pandas.testing import assert_frame_equal

import numpy as np
from anndata import AnnData


class TestTrainer(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataloader = DataLoader()
        self.taskloader = TaskLoader()
        self.trainer = Trainer()
    
    def test_one_vs_all_split(self):
        ann = self.dataloader.get_simple_protein_metabolite_dataframe()
        gen = self.taskloader.prepare_task(ann, Tier.TIER0).build(Strategy.ONE_VS_ALL)
        tf = next(gen)

        X_train, X_test, y_train, y_test = self.trainer.do_train_test_split(
            tf,
            strategy=Strategy.ONE_VS_ALL
        )

        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])
        

