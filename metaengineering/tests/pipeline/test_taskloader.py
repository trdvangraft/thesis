import unittest

from src.pipeline.dataloader import DataLoader
from src.pipeline.taskloader import TaskLoader

import pandas as pd
from pandas.testing import assert_frame_equal

import numpy as np


class TestTaskLoader(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataloader = DataLoader()
        self.taskloader = TaskLoader()

    def test_metabolite_concentration_prediction_simple(self):
        data = pd.DataFrame(data=np.arange(10).reshape((2, 5)), index=['GENOTYPE_1', 'GENOTYPE_2'], columns=[
            'GEN1', 'GEN2', 'GEN3', 'atp', 'adp'])

        df = self.taskloader.get_simple(data)

        # we expect the data the end up be genetype X gen X metabolite
        expected = pd.DataFrame(data={
            'GEN1': [0, 5, 0, 5],
            'GEN2': [1, 6, 1, 6],
            'GEN3': [2, 7, 2, 7],
            'metabolite_id': ['atp', 'atp', 'adp', 'adp'],
            'metabolite_concentration': [3, 8, 4, 9],
        }, index=['GENOTYPE_1', 'GENOTYPE_2', 'GENOTYPE_1', 'GENOTYPE_2'])

        assert_frame_equal(df, expected)

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

    def test_simple_single_happy_flow(self):
        df = self.dataloader.get_simple_protein_metabolite_dataframe()
        df = self.taskloader.get_simple(df)

        self.assertTupleEqual(df.shape, (2053, 728))
        self.assertEqual(len(df['metabolite_id'].unique()), 50)
        self.assertEqual(len(df.index.unique()), 96)
