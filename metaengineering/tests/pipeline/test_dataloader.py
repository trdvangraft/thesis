import unittest
from unittest.mock import Mock, patch

import pandas as pd

import numpy as np

from metaengineering.src.pipeline.dataloader import DataLoader, DataLoaderConfig


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
    
    def test_precursor_metabolite_filtering(self):
        config = DataLoaderConfig(
            additional_filters=[self.dataloader.data_factory.filters.is_precursor]
        )

        print(config)
        df = self.dataloader.get_dataframe(config)
        self.assertTupleEqual(df.obs.shape, (96, 12))
        self.assertListEqual(
            df.obs.columns.to_list(), 
            ['g6p;g6p-B', 'g6p;f6p;g6p-B', 'f6p', 'dhap', '3pg;2pg', 'pep', 'pyr', 'r5p', 'e4p', 'accoa', 'akg', 'oaa']
        )
    
    def test_transformer_protein_logfc(self):
        config = DataLoaderConfig(
            additional_filters=[self.dataloader.data_factory.filters.is_precursor],
            additional_transforms=[self.dataloader.data_factory.transformer.log_fold_change_protein]
        )

        df = self.dataloader.get_dataframe(config)
        self.assertTupleEqual(df.obs.shape, (95, 12))
        self.assertTupleEqual(df.X.shape, (95, 726))
    
    def test_ppi_network(self):
        config = DataLoaderConfig(
            additional_frames=[
                self.dataloader.data_factory.loaders.interaction_frame,
            ],
            additional_filters=[
                self.dataloader.data_factory.filters.is_precursor
            ],
            additional_transforms=[
                self.dataloader.data_factory.transformer.log_fold_change_protein,
                self.dataloader.data_factory.transformer.ppi_coo_matrix,
            ]
        )

        df = self.dataloader.get_dataframe(config)
        self.assertTupleEqual(df.obs.shape, (95, 12))
        self.assertTupleEqual(df.varp['ppi'].shape, (704, 704))
    
    def test_ppi_network_filter(self):
        config = DataLoaderConfig(
            additional_frames=[
                self.dataloader.data_factory.loaders.interaction_frame,
            ],
            additional_filters=[
                self.dataloader.data_factory.filters.is_precursor,
                self.dataloader.data_factory.filters.has_at_least_n_interaction,
            ],
            additional_transforms=[
                self.dataloader.data_factory.transformer.log_fold_change_protein,
                self.dataloader.data_factory.transformer.ppi_coo_matrix,
            ]
        )

        df = self.dataloader.get_dataframe(config)
        self.assertTupleEqual(df.obs.shape, (95, 12))
        self.assertTupleEqual(df.varp['ppi'].shape, (410, 410))

    
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

            
