from src.pipeline.frame.cache import FrameCache

from typing import List

import pandas as pd


class FrameFilters:
    def __init__(self, 
        frame_cache: FrameCache,
    ) -> None:
        self.frame_cache = frame_cache

    def is_in_genotype(self):
        for key, cf in self.frame_cache.get_all_frames():
            if type(cf) == pd.DataFrame:
                self._is_in_genotype(key, cf)
            elif type(cf) == dict:
                for key1, cf1 in cf.items():
                    self._is_in_genotype(key1, cf1)

    def is_precursor(self):
        precursor_metabolite_ids = [
            'g6p;g6p-B', 'g6p;f6p;g6p-B', 'f6p', 'dhap', '3pg;2pg',
            'pep', 'pyr', 'r5p', 'e4p', 'accoa', 'akg', 'oaa',
        ]

        obs = self.frame_cache.get_frame('metabolites')
        obs = obs[precursor_metabolite_ids]
        self.frame_cache.update_frame('metabolites', obs)
    
    def parse_config(self, function_names: List[str]):
        string_to_function = {
            "is_precursor": self.is_precursor,
            "has_at_least_n_interaction": self.has_at_least_n_interaction,
            "is_in_genotype": self.is_in_genotype,
        }
        return [string_to_function.get(function_name) for function_name in function_names]
    
    def has_at_least_n_interaction(self):
        # TODO: we need to encode this into a parameter
        n = 7 
        # Assume that the transformation to the adj matrix worked out
        cf = self.frame_cache.get_frame('ppi')
        # sum over all the rows
        cf = cf.loc[cf.sum(axis=1) > n]
        assert cf.sum(axis=1).min() > n
        self.frame_cache.update_frame('ppi', cf)
    
    def _is_in_genotype(self, key: str, cf: pd.DataFrame):
        if 'KO_ORF' not in cf.index.names:
            return

        _df = cf[cf.index.get_level_values('KO_ORF').isin(self.frame_cache.get_frame('metabolites').index)]
        self.frame_cache.update_frame(key, _df)
        