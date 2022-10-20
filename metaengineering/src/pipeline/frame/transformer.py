from io import StringIO
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
    
    def ppi(self):
        _df = self.frame_cache.get_frame('ppi')
        _df['interaction_network'] = _df['interaction_network'].apply(self._to_dataframe)
        _df['adj_matrix'] = _df['interaction_network'].apply(self._to_adj_matrix)
        _df['node_labels'] = _df['interaction_network'].apply(self._get_node_labels)

        self.frame_cache.update_frame('ppi', _df)
    
    def _get_node_labels(x: pd.DataFrame) -> np.array:
        name_a, name_b = 'stringId_A', 'stringId_B'
        return np.unique(pd.concat([x[name_a], x[name_b]]))

    def _to_dataframe(x):
        tsv_string = StringIO(x)
        df = pd.read_csv(tsv_string, sep="\t", names=[
            'stringId_A', 'stringId_B', 'preferredName_A', 'preferredName_B', 
            'ncbiTaxonId', 'score', 'nscore', 'fscore', 'pscore', 'ascore',
            'escore', 'dscore', 'tscore'
        ])
        df = df.drop_duplicates(subset=['preferredName_A', 'preferredName_B'], keep='last')

        df['stringId_A'] = df['stringId_A'].map(lambda x: x.strip('4932.'))
        df['stringId_B'] = df['stringId_B'].map(lambda x: x.strip('4932.'))

        return df[['stringId_A', 'stringId_B', 'score']]

    def _to_adj_matrix(x: pd.DataFrame) -> np.matrix:
        def get_idx(name: str) -> np.array:
            return [np.where(unique_labels == elem)[0][0] for _, elem in x[name].iteritems()]

        name_a, name_b = 'stringId_A', 'stringId_B'
        unique_labels = np.unique(pd.concat([x[name_a], x[name_b]]))
        graph_size = len(unique_labels)

        return coo_matrix((x['score'].values, (get_idx(name_a), get_idx(name_b))), shape=(graph_size, graph_size))
    
    def _apply_fc(self, _df: pd.DataFrame):
        _df.loc[:, _df.columns] = np.log(_df.loc[:, _df.columns])      
        _df = _df - _df.loc['WT'].values.squeeze()
        _df = _df.drop('WT', axis=0)
        return _df