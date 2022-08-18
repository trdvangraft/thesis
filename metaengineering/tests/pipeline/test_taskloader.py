import unittest
from metaengineering.src.settings import Strategy

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
            gen = self.taskloader.prepare_task(ann).build(Strategy.ALL)
            tf: TaskFrame = next(gen)

            self.assertListEqual(tf.x.index.unique(0).to_list(), ['GENOTYPE_1', 'GENOTYPE_2'])
            self.assertListEqual(tf.x.index.unique(1).to_list(), ['adp', 'atp'])
            self.assertListEqual(tf.x.index.unique(2).to_list(), ['GEN1', 'GEN2', 'GEN3'])
            self.assertTupleEqual(tf.x.shape, (12, 1))
        
        with self.subTest('genotype strategy'):
            gen = self.taskloader.prepare_task(ann).build(Strategy.GENOTYPE_CENTRIC)
            genotypes = ['GENOTYPE_1', 'GENOTYPE_2']
            for frame, genotype in zip(gen, genotypes):
                with self.subTest(f'\t {genotype}'):
                    self.assertEqual(frame.frame_name, genotype)
                    self.assertListEqual(frame.x.index.unique(0).to_list(), ['adp', 'atp'])
                    self.assertListEqual(frame.x.index.unique(1).to_list(), ['GEN1', 'GEN2', 'GEN3'])

        with self.subTest('metabolite strategy'):
            gen = self.taskloader.prepare_task(ann).build(Strategy.METABOLITE_CENTRIC)
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
            gen = self.taskloader.prepare_task(ann).build(Strategy.ALL)
            tf: TaskFrame = next(gen)

            self.assertListEqual(tf.x.index.unique(0).to_list(), ['GENOTYPE_1', 'GENOTYPE_2'])
            self.assertListEqual(tf.x.index.unique(1).to_list(), ['adp', 'atp'])
            self.assertListEqual(tf.x.index.unique(2).to_list(), ['GEN1', 'GEN3'])
            self.assertTupleEqual(tf.x.shape, (5, 1))
        
        with self.subTest('genotype strategy'):
            gen = self.taskloader.prepare_task(ann).build(Strategy.GENOTYPE_CENTRIC)

            genotype_one_frame: TaskFrame = next(gen)
            genotype_two_frame: TaskFrame = next(gen)

            self.assertListEqual(genotype_one_frame.x.index.unique(0).to_list(), ['adp', 'atp'])
            self.assertListEqual(genotype_one_frame.x.index.unique(1).to_list(), ['GEN1', 'GEN3'])

            self.assertListEqual(genotype_two_frame.x.index.unique(0).to_list(), ['adp'])
            self.assertListEqual(genotype_two_frame.x.index.unique(1).to_list(), ['GEN3'])

        with self.subTest('metabolite strategy'):
            gen = self.taskloader.prepare_task(ann).build(Strategy.METABOLITE_CENTRIC)

            metabolite_one_frame: TaskFrame = next(gen)
            metabolite_two_frame: TaskFrame = next(gen)

            self.assertListEqual(metabolite_one_frame.x.index.unique(0).to_list(), ['GENOTYPE_1', 'GENOTYPE_2'])
            self.assertListEqual(metabolite_one_frame.x.index.unique(1).to_list(), ['GEN1', 'GEN3'])

            self.assertListEqual(metabolite_two_frame.x.index.unique(0).to_list(), ['GENOTYPE_1'])
            self.assertListEqual(metabolite_two_frame.x.index.unique(1).to_list(), ['GEN1', 'GEN3'])

    def test_get_simple_removes_nan(self):
        data = pd.DataFrame(
            data=[[0.7, np.NAN, 0.2], [0.3, 0.5, np.NAN]],
            index=['GENOTYPE_1', 'GENOTYPE_2'],
            columns=['GEN1', 'atp', 'adp']
        )

        df = self.taskloader.get_simple(data)

        # we expect the data the end up be genetype X gen X metabolite
        expected = pd.DataFrame(data={
            'GEN1': [0.3, 0.7],
            'metabolite_id': ['atp', 'adp'],
            'metabolite_concentration': [0.5, 0.2],
        }, index=['GENOTYPE_2', 'GENOTYPE_1'])

        assert_frame_equal(df, expected)
    
    def test_dataset_all_strategy(self):
        ann = self.dataloader.get_simple_protein_metabolite_dataframe()
        gen = self.taskloader.prepare_task(ann).build(Strategy.ALL)

        number_of_frames = sum(1 for _ in gen)
        self.assertEqual(number_of_frames, 1)

    def test_dataset_metabolite_strategy(self):
        ann = self.dataloader.get_simple_protein_metabolite_dataframe()
        gen = self.taskloader.prepare_task(ann).build(Strategy.METABOLITE_CENTRIC)

        number_of_frames = sum(1 for _ in gen)
        self.assertEqual(number_of_frames, 50)

    def test_dataset_genotype_strategy(self):
        ann = self.dataloader.get_simple_protein_metabolite_dataframe()
        gen = self.taskloader.prepare_task(ann).build(Strategy.GENOTYPE_CENTRIC)

        number_of_frames = sum(1 for _ in gen)
        self.assertEqual(number_of_frames, 96)
