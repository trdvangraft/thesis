from src.pipeline.frame.cache import FrameCache

import pandas as pd


class FrameFilters:
    def __init__(self, 
        frame_cache: FrameCache,
    ) -> None:
        self.frame_cache = frame_cache

    def is_in_genotype(self):
        for key, cf in self.frame_cache.get_all_frames():
            if type(cf) == pd.DataFrame:
                _df = cf[cf.index.get_level_values('KO_ORF').isin(self.frame_cache.get_frame('metabolites').index)]
                self.frame_cache.update_frame(key, _df)
            elif type(cf) == dict:
                for key1, cf1 in cf.items():
                    _df = cf1[cf1.index.get_level_values('KO_ORF').isin(self.frame_cache.get_frame('metabolites').index)]
                    self.frame_cache.update_frame(key1, _df)

    def is_precursor(self):
        precursor_metabolite_ids = [
            'g6p;g6p-B', 'g6p;f6p;g6p-B', 'f6p', 'dhap', '3pg;2pg',
            'pep', 'pyr', 'r5p', 'e4p', 'accoa', 'akg', 'oaa',
        ]

        obs = self.frame_cache.get_frame('metabolites')
        obs = obs[precursor_metabolite_ids]
        self.frame_cache.update_frame('metabolites', obs)