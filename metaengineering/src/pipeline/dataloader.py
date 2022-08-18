from typing import Callable, Dict, List, OrderedDict, Union
import pandas as pd
import numpy as np

import pyreadr

from src.settings import DataOrientation

from anndata import AnnData


class DataFactory:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.frame_cache: Dict[str, pd.DataFrame] = {}
        self.current_frame: AnnData = None

    def load_frames(self, frames: List[Callable]):
        for frame_loader in frames:
            frame_loader()
        return self
    
    def get_basic_frame(self):
        self.frame_cache['metabolites'] = self.load_csv(f'{self.root_dir}metabolites_dataset.data_prep.tsv')
        self.frame_cache['proteins'] = self.load_csv(f'{self.root_dir}proteins_dataset.data_prep.tsv')

        return self
    
    def get_protein_expression_frame(self):
        self.frame_cache['protein_expression'] = self.load_r(f'{self.root_dir}proteins.matrix.sva.0.5.1.FC.RData')
        return self

    def transform(self, transforms: List[Callable]):
        for transform in transforms:
            transform()
        return self
    
    def transform_x(self):
        self.frame_cache['proteins'] = self.frame_cache['proteins'] \
            .groupby(by=['KO_ORF', 'ORF'])['value'].mean() \
            .to_frame().pivot_table(index='KO_ORF', columns='ORF', values='value')

        self.frame_cache['proteins'] = self.filter_is_in_genotype('proteins', 'KO_ORF')
        
        return self
    
    def transform_obs(self):
        self.frame_cache['metabolites'] = self.frame_cache['metabolites'] \
            .pivot_table(index='genotype', columns='metabolite_id', values='value')
        
        self.frame_cache['metabolites'] = self._get_frame('metabolites').rename_axis('KO_ORF')
        
        return self
    
    def transform_protein(self):
        target_columns = ['logFC', 'p.value', 'p.value_BH', 'p.value_bonferroni']
        self.frame_cache['protein_expression'] = self.filter_is_in_genotype('protein_expression', 'KO')
        self.frame_cache['protein_expression'] = { 
            key: self._get_frame('protein_expression').pivot_table(values=key, index='ORF', columns='KO') for key in target_columns
        }
        return self

    def combine_x_and_obs(self):
        self.current_frame = AnnData(
            X=self.frame_cache['proteins'], 
            obs=self.frame_cache['metabolites']
        )
        
        return self
    
    def combine_proteins(self):      
        for key, item in self._get_frame('protein_expression').items():
            self.current_frame.varm[key] = item
        
        return self
    
    def build(self):
        df = self.current_frame.copy()
        self.current_frame = None
        return df

    def filter_is_in_genotype(self, frame_name: str, column_name: str):
        cf = self._get_frame(frame_name)

        if column_name in cf.columns:
            return cf[cf[column_name].isin(self._get_frame('metabolites').index)]
        elif column_name == cf.index.name:
            return cf[cf.index.isin(self._get_frame('metabolites').index)]
        else:
            # TODO: We might want to throw an exception here if the column is not in the columns
            pass
    
    def load_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, delimiter='\t')
    
    def load_r(self, path: str):
        dataframe_dict: OrderedDict = pyreadr.read_r(path)
        return dataframe_dict[list(dataframe_dict.keys())[0]]
    
    def _get_frame(self, name) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        return self.frame_cache[name]


    

class DataLoader:
    DATA_FOLDER = './metaengineering/data/'

    def __init__(self) -> None:
        self.data_factory = DataFactory(DataLoader.DATA_FOLDER)

    def get_simple_protein_metabolite_dataframe(self):
        """
        This dataframe is the simplest way for predicting the metabolite concentrations
        Produce a dataframe with the genotype as key and columns concatenated (genotype X (proteins + metabolites))

        Averages the repeated experiments of the raw protein dataset
        """
        return self.data_factory.get_basic_frame() \
            .transform(transforms=[
                self.data_factory.transform_obs,
                self.data_factory.transform_x
            ]) \
            .combine_x_and_obs() \
            .build()
    
    def get_simple_diff_expr_dataframe(self):
        return self.data_factory.load_frames(
            frames=[
                self.data_factory.get_basic_frame,
                self.data_factory.get_protein_expression_frame
            ]
        ) \
            .transform(transforms=[
                self.data_factory.transform_obs,
                self.data_factory.transform_x,
                self.data_factory.transform_protein
            ]) \
            .combine_x_and_obs() \
            .combine_proteins() \
            .build()

    @staticmethod
    def _get_metabolite_names():
        """
        We need to extract metabolite names from the raw metabolites table
        """
        raw_metabolites = pd.read_csv(
            f'{DataLoader.DATA_FOLDER}metabolites_dataset.data_prep.tsv', delimiter='\t')

        return raw_metabolites['metabolite_id'].unique()
