from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, OrderedDict, Union
import pandas as pd
import numpy as np

import pyreadr


from anndata import AnnData

class FrameCache:
    def __init__(self) -> None:
        self.frame_cache: Dict[str, Union[pd.DataFrame, FrameCache]] = {}

    def insert_frame(self, name: str, frame: pd.DataFrame):
        self.frame_cache[name] = frame.copy()
    
    def get_frame(self, name, default: Any=None):
        return self.frame_cache.get(name, default)
    
    def update_frame(self, name: str, frame: Union[pd.DataFrame, Any]):
        if type(frame) == pd.DataFrame:
            self.frame_cache[name] = frame.copy()
        elif type(frame) == FrameCache:
            self.frame_cache[name] = frame
    
    def get_all_frames(self):
        return self.frame_cache.items()
    
    def contains(self, name: str):
        return name in self.frame_cache

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
    
    def _load_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, delimiter='\t')
    
    def _load_r(self, path: str):
        dataframe_dict: OrderedDict = pyreadr.read_r(path)
        return dataframe_dict[list(dataframe_dict.keys())[0]]

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
    
    def _apply_fc(self, _df: pd.DataFrame):
        _df.loc[:, _df.columns] = np.log(_df.loc[:, _df.columns])      
        _df = _df - _df.loc['WT'].values.squeeze()
        _df = _df.drop('WT', axis=0)
        return _df

class DataFactory:
    def __init__(self, root_dir: str) -> None:
        self.frame_cache: FrameCache = FrameCache()
        self._transformer = FrameTransformers(self.frame_cache)
        self._loaders = FrameLoaders(self.frame_cache, root_dir)
        self._filters = FrameFilters(self.frame_cache)

    @property
    def loaders(self):
        return self._loaders
    
    @property
    def transformer(self):
        return self._transformer
    
    @property
    def filters(self):
        return self._filters

    def load(self, frames: List[Callable]):
        for frame_loader in frames:
            frame_loader()
        return self

    def transform(self, transforms: List[Callable]):
        for transform in transforms:
            transform()
        return self
    
    def filter(self, filters: List[Callable] = []):
        for filter in filters:
            filter()
        return self

    def build(self):
        self.current_frame = AnnData(
            X=self.frame_cache.get_frame('proteins'), 
            obs=self.frame_cache.get_frame('metabolites')
        )

        if self.frame_cache.contains('protein_expression'):
            for key, item in self.frame_cache.get_frame('protein_expression', default={}).get_all_frames():
                self.current_frame.varm[key] = item

        df = self.current_frame.copy()
        self.current_frame = None
        return df


@dataclass
class DataLoaderConfig:
    additional_frames: List = field(default_factory=list)
    additional_transforms: List = field(default_factory=list)
    additional_filters: List = field(default_factory=list)
    

class DataLoader:
    DATA_FOLDER = './metaengineering/data/training/'

    def __init__(self) -> None:
        self.data_factory = DataFactory(DataLoader.DATA_FOLDER)

    def get_dataframe(
        self,
        config: DataLoaderConfig = DataLoaderConfig()
    ):
        """
        This dataframe is the simplest way for predicting the metabolite concentrations
        Produce a dataframe with the genotype as key and columns concatenated (genotype X (proteins + metabolites))

        Averages the repeated experiments of the raw protein dataset
        """
        df = self.data_factory
        print(config)

        return df \
            .load(frames=[
                    df.loaders.basic_frame
                ] + config.additional_frames
            ) \
            .transform(transforms=[
                    df.transformer.metabolites,
                    df.transformer.proteins
                ] + config.additional_transforms
            ) \
            .filter(filters=[
                    df.filters.is_in_genotype
                ] + config.additional_filters
            ) \
            .build()

    def get_simple_protein_metabolite_dataframe(
        self
    ):
        """
        This dataframe is the simplest way for predicting the metabolite concentrations
        Produce a dataframe with the genotype as key and columns concatenated (genotype X (proteins + metabolites))

        Averages the repeated experiments of the raw protein dataset
        """
        return self.get_dataframe()

    
    def get_simple_diff_expr_dataframe(self):
        df = self.data_factory

        return df \
            .load(frames=[
                    df.loaders.basic_frame,
                    df.loaders.protein_expression_frame
                ]
            ) \
            .transform(transforms=[
                df.transformer.metabolites,
                df.transformer.proteins,
                df.transformer.protein_expression
            ]) \
            .filter(filters=[
                df.filters.is_in_genotype
            ]) \
            .build()

    @staticmethod
    def _get_metabolite_names():
        """
        We need to extract metabolite names from the raw metabolites table
        """
        raw_metabolites = pd.read_csv(
            f'{DataLoader.DATA_FOLDER}metabolites_dataset.data_prep.tsv', delimiter='\t')

        return raw_metabolites['metabolite_id'].unique()
