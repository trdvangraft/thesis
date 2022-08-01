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
