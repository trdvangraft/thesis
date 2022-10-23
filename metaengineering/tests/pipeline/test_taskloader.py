import unittest
from metaengineering.src.pipeline.taskloader import TaskLoaderConfig
from metaengineering.src.settings.strategy import Strategy
from metaengineering.src.settings.tier import Tier

from src.pipeline.dataloader import DataLoader
from src.pipeline.taskloader import TaskFrame, TaskLoader

import pandas as pd
from pandas.testing import assert_frame_equal

import numpy as np
from anndata import AnnData


class TestTaskLoader(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataloader = DataLoader()
        self.taskloader = TaskLoader()

    def test_formats_taskframes_correctly(self):
        data = pd.DataFrame(data=np.arange(6).reshape((2, 3)), index=['GENOTYPE_1', 'GENOTYPE_2'], columns=[
            'GEN1', 'GEN2', 'GEN3']).rename_axis(index='KO_ORF', columns='ORF')
        obs = pd.DataFrame(
            data=np.arange(4).reshape((2, 2)), 
            index=['GENOTYPE_1', 'GENOTYPE_2'], 
            columns=['atp', 'adp']
        ).rename_axis('KO_ORF')

        ann = AnnData(X=data, obs=obs)

        with self.subTest('Stragey all'):
            gen = self.taskloader.prepare_task(ann, Tier.TIER1).build(Strategy.ALL)
            tf: TaskFrame = next(gen)

            self.assertListEqual(tf.x.index.unique(0).to_list(), ['GENOTYPE_1', 'GENOTYPE_2'])
            self.assertListEqual(tf.x.index.unique(1).to_list(), ['adp', 'atp'])
            self.assertListEqual(tf.x.index.unique(2).to_list(), ['GEN1', 'GEN2', 'GEN3'])
            self.assertTupleEqual(tf.x.shape, (12, 1))
    
        with self.subTest('metabolite strategy'):
            gen = self.taskloader.prepare_task(ann, Tier.TIER1).build(Strategy.METABOLITE_CENTRIC)
            metabolites = ['adp', 'atp']
            for frame, metabolite in zip(gen, metabolites):
                with self.subTest(f'\t {metabolite}'):
                    self.assertEqual(frame.frame_name, metabolite)
                    self.assertListEqual(frame.x.index.unique(0).to_list(), ['GENOTYPE_1', 'GENOTYPE_2'])
                    self.assertListEqual(frame.x.index.unique(1).to_list(), ['GEN1', 'GEN2', 'GEN3'])
    
    def test_removes_nan(self):
        data = pd.DataFrame(
            data=[[0.2, np.NAN, 0.5], [np.NAN, np.NAN, 0.7]], 
            index=['GENOTYPE_1', 'GENOTYPE_2'], 
            columns=['GEN1', 'GEN2', 'GEN3']
        ).rename_axis(index='KO_ORF', columns='ORF')
        obs = pd.DataFrame(
            data=[[1.2, 1.2], [np.NAN, 0.5]], 
            index=['GENOTYPE_1', 'GENOTYPE_2'], 
            columns=['atp', 'adp']
        ).rename_axis('KO_ORF')

        ann = AnnData(X=data, obs=obs)

        with self.subTest('Stragey all'):
            gen = self.taskloader.prepare_task(ann, Tier.TIER1).build(Strategy.ALL)
            tf: TaskFrame = next(gen)

            self.assertListEqual(tf.x.index.unique(0).to_list(), ['GENOTYPE_1', 'GENOTYPE_2'])
            self.assertListEqual(tf.x.index.unique(1).to_list(), ['adp', 'atp'])
            self.assertListEqual(tf.x.index.unique(2).to_list(), ['GEN1', 'GEN3'])
            self.assertTupleEqual(tf.x.shape, (5, 1))

        with self.subTest('metabolite strategy'):
            gen = self.taskloader.prepare_task(ann, Tier.TIER1).build(Strategy.METABOLITE_CENTRIC)

            metabolite_one_frame: TaskFrame = next(gen)
            metabolite_two_frame: TaskFrame = next(gen)

            self.assertListEqual(metabolite_one_frame.x.index.unique(0).to_list(), ['GENOTYPE_1', 'GENOTYPE_2'])
            self.assertListEqual(metabolite_one_frame.x.index.unique(1).to_list(), ['GEN1', 'GEN3'])

            self.assertListEqual(metabolite_two_frame.x.index.unique(0).to_list(), ['GENOTYPE_1'])
            self.assertListEqual(metabolite_two_frame.x.index.unique(1).to_list(), ['GEN1', 'GEN3'])
    
    def test_dataset_all_strategy(self):
        ann = self.dataloader.get_simple_protein_metabolite_dataframe()
        gen = self.taskloader.prepare_task(ann, Tier.TIER1).build(Strategy.ALL)

        number_of_frames = sum(1 for _ in gen)
        self.assertEqual(number_of_frames, 1)

    def test_dataset_metabolite_strategy(self):
        ann = self.dataloader.get_simple_protein_metabolite_dataframe()
        gen = self.taskloader.prepare_task(ann, Tier.TIER1).build(Strategy.METABOLITE_CENTRIC)

        number_of_frames = sum(1 for _ in gen)
        self.assertEqual(number_of_frames, 50)

    def test_base_metabolite_strategy(self):
        ann = self.dataloader.get_simple_protein_metabolite_dataframe()
        gen = self.taskloader.prepare_task(ann, Tier.TIER0).build(Strategy.METABOLITE_CENTRIC)

        number_of_frames = sum(1 for _ in gen)
        self.assertEqual(number_of_frames, 50)
    
    def test_data_throttle(self):
        ann = self.dataloader.get_simple_protein_metabolite_dataframe()
        print(ann.X.shape)

        config = TaskLoaderConfig(data_throttle=.5)
        gen = self.taskloader.prepare_task(ann, Tier.TIER0).build(Strategy.ALL, config)

        tf = next(gen)
        self.assertTupleEqual(tf.x.shape, (1021, 726))
    
    def test_one_vs_all_strategy(self):
        ann = self.dataloader.get_simple_protein_metabolite_dataframe()
        gen = self.taskloader.prepare_task(ann, Tier.TIER0).build(Strategy.ONE_VS_ALL)

        number_of_frames = sum(1 for _ in gen)
        self.assertEqual(number_of_frames, 50)

