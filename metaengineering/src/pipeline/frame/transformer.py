from io import StringIO
from typing import List
from src.pipeline.frame.cache import FrameCache

import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix

class FrameTransformers:
    def __init__(self, 
        frame_cache: FrameCache,
    ) -> None:
        self.frame_cache = frame_cache

    def proteins(self):
        _df = self.frame_cache.get_frame('proteins') \
            .groupby(by=['KO_ORF', 'ORF'])['value'].mean() \
            .to_frame().pivot_table(index='KO_ORF', columns='ORF', values='value')

        self.frame_cache.update_frame('proteins', _df)
    
    def metabolites(self):
        _df = self.frame_cache.get_frame('metabolites') \
            .pivot_table(index='genotype', columns='metabolite_id', values='value') \
            .rename_axis('KO_ORF')

        self.frame_cache.update_frame('metabolites', _df)
    
    def log_fold_change_protein(self):
        for frame_name in ['proteins', 'metabolites']:
            _df = self.frame_cache.get_frame(frame_name)
            self.frame_cache.update_frame(frame_name, self._apply_fc(_df))\
    
    def protein_expression(self):
        target_columns = ['logFC', 'p.value', 'p.value_BH', 'p.value_bonferroni']
        fc = FrameCache()

        for key in target_columns:
            fc.insert_frame(
                key,
                self.frame_cache.get_frame('protein_expression').pivot_table(values=key, index='ORF', columns='KO').rename_axis('KO_ORF')
            )

        self.frame_cache.update_frame('protein_expression', fc)

    def ppi_coo_matrix(self):
        _df = self.frame_cache.get_frame('ppi')
        _df['interaction_network'] = _df['interaction_network'].apply(_to_dataframe)

        _df = pd.concat(_df['interaction_network'].values)
        _df = self._ppi_coo_matrix(_df)
        self.frame_cache.update_frame('ppi', _df)

    def parse_config(self, function_names: List[str]):
        string_to_function = {
            "ppi_coo_matrix": self.ppi_coo_matrix,
            "protein_expression": self.protein_expression,
            "log_fold_change_protein": self.log_fold_change_protein,
            "metabolites": self.metabolites,
            "proteins": self.proteins
        }
        return [string_to_function.get(function_name) for function_name in function_names]
    
    def _ppi_coo_matrix(self, df):
        def get_idx(name: str) -> np.array:
            return [np.where(proteins == elem)[0][0] for _, elem in df[name].items()]

        name_a, name_b = 'stringId_A', 'stringId_B'

        proteins = self.frame_cache.get_frame('proteins').columns.unique(0)
        df = df[(df[name_a].isin(proteins)) & (df[name_b].isin(proteins))]

        adj_matrix = np.zeros(shape=(len(proteins), len(proteins)))

        name_a_idx = get_idx(name_a)
        name_b_idx = get_idx(name_b)

        for i, j in zip(name_a_idx, name_b_idx):
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

        df = pd.DataFrame(
            data=adj_matrix,
            columns=proteins,
            index=proteins,
        )

        filtering = 30

        df = df.loc[
            (df.sum(axis=1) >= filtering),
            (df.sum(axis=0) >= filtering)
        ]

        return df

    def _apply_fc(self, _df: pd.DataFrame):
        _df.loc[:, _df.columns] = np.log2(_df.loc[:, _df.columns])      
        _df = _df - _df.loc['WT'].values.squeeze()
        _df = _df.drop('WT', axis=0)
        return _df

def _to_dataframe(x):
    tsv_string = StringIO(x)
    name_a, name_b = 'stringId_A', 'stringId_B'
    df = pd.read_csv(tsv_string, sep="\t", names=[
        'stringId_A', 'stringId_B', 'preferredName_A', 'preferredName_B', 
        'ncbiTaxonId', 'score', 'nscore', 'fscore', 'pscore', 'ascore',
        'escore', 'dscore', 'tscore'
    ])
    df = df.drop_duplicates(subset=['preferredName_A', 'preferredName_B'], keep='last')

    df[name_a] = df[name_a].map(lambda x: x.strip('4932.'))
    df[name_b] = df[name_b].map(lambda x: x.strip('4932.'))
    return df

