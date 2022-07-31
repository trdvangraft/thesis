import unittest
from unittest.mock import Mock, patch
from metaengineering.src.settings import DataOrientation

from src.dataloader import DataLoader

import pandas as pd
from pandas.testing import assert_frame_equal

import numpy as np


class TestDataloader(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataloader = DataLoader()

    @patch('src.dataloader.pd.read_csv')
    def test_dataloader_merges_correctly(self, mock_read_csv: Mock):
        def data_frame_return(path, delimiter):
            if 'metabolite' in path:
                return pd.DataFrame(data={
                    'genotype': ['GENOTYPE_1', 'GENOTYPE_1', 'GENOTYPE_2', 'GENOTYPE_2'],
                    'metabolite_id': ['METABOLITE1', 'METABOLITE2', 'METABOLITE1', 'METABOLITE2'],
                    'value': [0.1, 0.2, 0.3, 0.4]
                })
            elif 'protein' in path:
                return pd.DataFrame(data={
                    'KO_ORF': ['GENOTYPE_1', 'GENOTYPE_1', 'GENOTYPE_2', 'GENOTYPE_2'],
                    'ORF': ['GEN1', 'GEN2', 'GEN1', 'GEN2'],
                    'value': [0.1, 0.2, 0.3, 0.4]
                })

        mock_read_csv.side_effect = data_frame_return

        df = self.dataloader.get_simple_protein_metabolite_dataframe()

        mock_read_csv.assert_called()
        self.assertListEqual(df.index.to_list(), ['GENOTYPE_1', 'GENOTYPE_2'])
        self.assertListEqual(df.columns.to_list(), [
                             'GEN1', 'GEN2', 'METABOLITE1', 'METABOLITE2'])
        self.assertTupleEqual(df.shape, (2, 4))

    def test_metabolite_concentration_prediction_simple(self):
        data = pd.DataFrame(data=np.arange(10).reshape((2, 5)), index=['GENOTYPE_1', 'GENOTYPE_2'], columns=[
            'GEN1', 'GEN2', 'GEN3', 'atp', 'adp'])

        result = self.dataloader.get_simple(
            data)

        # we expect the data the end up be genetype X gen X metabolite
        expected = pd.DataFrame(data={
            'GEN1': [0, 5, 0, 5],
            'GEN2': [1, 6, 1, 6],
            'GEN3': [2, 7, 2, 7],
            'metabolite_id': ['atp', 'atp', 'adp', 'adp'],
            'metabolite_concentration': [3, 8, 4, 9]
        }, index=['GENOTYPE_1', 'GENOTYPE_2', 'GENOTYPE_1', 'GENOTYPE_2'])

        assert_frame_equal(result, expected)

    def test_get_simple_removes_nan(self):
        data = pd.DataFrame(
            data=[[0.7, np.NAN, 0.2], [0.3, 0.5, np.NAN]],
            index=['GENOTYPE_1', 'GENOTYPE_2'],
            columns=['GEN1', 'atp', 'adp']
        )

        result = self.dataloader.get_simple(
            data)

        # we expect the data the end up be genetype X gen X metabolite
        expected = pd.DataFrame(data={
            'GEN1': [0.3, 0.7],
            'metabolite_id': ['atp', 'adp'],
            'metabolite_concentration': [0.5, 0.2]
        }, index=['GENOTYPE_2', 'GENOTYPE_1'])

        assert_frame_equal(result, expected)

    def test_simple_single_happy_flow(self):
        df = self.dataloader.get_simple_protein_metabolite_dataframe()
        df = self.dataloader.get_simple(df)

        self.assertTupleEqual(df.shape, (2053, 728))
        self.assertEqual(len(df['metabolite_id'].unique()), 50)
        self.assertEqual(len(df.index.unique()), 96)
