import unittest
from unittest.mock import Mock, patch
from src.settings import DataOrientation

from src.pipeline.dataloader import DataLoader

import pandas as pd
from pandas.testing import assert_frame_equal

import numpy as np


class TestDataloader(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataloader = DataLoader()

    @patch('src.pipeline.dataloader.pd.read_csv')
    def test_dataloader_merges_correctly(self, mock_read_csv: Mock):
        mock_read_csv.side_effect = self._data_frame_return

        df = self.dataloader.get_simple_protein_metabolite_dataframe()

        mock_read_csv.assert_called()
        self.assertListEqual(df.obs_names.to_list(), ['GENOTYPE_1', 'GENOTYPE_2'])
        self.assertListEqual(df.obs_keys(), ['METABOLITE1', 'METABOLITE2'])
        self.assertListEqual(df.var_names.to_list(), ['GEN1', 'GEN2'])
        self.assertTupleEqual(df.X.shape, (2, 2))
    
    @patch('src.pipeline.dataloader.pyreadr.read_r')
    @patch('src.pipeline.dataloader.pd.read_csv')
    def test_differtianly_expressed_dataloader(self, mock_read_csv: Mock, mock_read_r: Mock):
        mock_read_csv.side_effect = self._data_frame_return
        mock_read_r.side_effect = self._data_pyreadr_proteins

        df = self.dataloader.get_simple_diff_expr_dataframe()
        
        mock_read_csv.assert_called()
        mock_read_r.assert_called()

        self.assertListEqual(df.obs_keys(), ['METABOLITE1', 'METABOLITE2'])
        self.assertListEqual(df.obs_names.to_list(), ['GENOTYPE_1', 'GENOTYPE_2'])
        self.assertListEqual(df.varm_keys(), ['logFC', 'p.value', 'p.value_BH', 'p.value_bonferroni'])
    
    def test_happy_flow(self):
        df = self.dataloader.get_simple_diff_expr_dataframe()

        self.assertTupleEqual(df.shape, (96, 726))
    
    def _data_frame_return(self, path, delimiter):
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
    
    def _data_pyreadr_proteins(self, path):
        return {
            'proteins.matrix.sva.0.5.1.FC': pd.DataFrame(data={
                'KO': ['GENOTYPE_1', 'GENOTYPE_1', 'GENOTYPE_2', 'GENOTYPE_2'],
                'ORF': ['GEN1', 'GEN2', 'GEN1', 'GEN2'],
                'logFC': np.ones(4),
                'p.value': np.ones(4),
                'p.value_BH': np.ones(4),
                'p.value_bonferroni': np.ones(4),
            })
        }
            
