from typing import OrderedDict, List

from src.pipeline.frame.cache import FrameCache

import pandas as pd
import pyreadr


class FrameLoaders:
    def __init__(self, 
        frame_cache: FrameCache,
        root_dir: str
    ) -> None:
        self.frame_cache = frame_cache
        self.root_dir = root_dir

    def basic_frame(self):
        self.frame_cache.insert_frame('metabolites', self._load_csv(f'{self.root_dir}metabolites_dataset.data_prep.tsv'))
        self.frame_cache.insert_frame('proteins', self._load_csv(f'{self.root_dir}proteins_dataset.data_prep.tsv'))
        return self
    
    def protein_expression_frame(self):
        self.frame_cache.insert_frame('protein_expression', self._load_r(f'{self.root_dir}proteins.matrix.sva.0.5.1.FC.RData'))
        return self
    
    def interaction_frame(self):
        self.frame_cache.insert_frame('ppi', self._load_json(f'{self.root_dir}tsv_records.json'))
        return self
    
    def go_frame(self):
        self.frame_cache.insert_frame('go', self._load_r(f'{self.root_dir}/GO.raw._load_.RData'))
        return self
    
    def exp_metadata_frame(self):
        self.frame_cache.insert_frame('exp_data', self._load_r(f'{self.root_dir}/exp_metadata._clean_.RData'))
        return self
    
    def parse_config(self, function_names: List[str]):
        string_to_function = {
            "exp_metadata_frame": self.exp_metadata_frame,
            "go_frame": self.go_frame,
            "interaction_frame": self.interaction_frame,
            "protein_expression_frame": self.protein_expression_frame,
            "basic_frame": self.basic_frame
        }
        return [string_to_function.get(function_name) for function_name in function_names]
    
    def _load_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, delimiter='\t')
    
    def _load_r(self, path: str):
        dataframe_dict: OrderedDict = pyreadr.read_r(path)
        return dataframe_dict[list(dataframe_dict.keys())[0]]
    
    def _load_json(self, path: str):
        return pd.read_json(path)