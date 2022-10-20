from dataclasses import dataclass, field
from typing import List
import pandas as pd

from src.pipeline.datafactory import DataFactory

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
    
    def get_go_dataframe(self):
        df = self.data_factory
        config = DataLoaderConfig(
            additional_frames=[
                df.loaders.protein_expression_frame,
                df.loaders.go_frame,
                df.loaders.exp_metadata_frame
            ],
            additional_transforms=[
                df.transformer.log_fold_change_protein
            ]
        )

        return self.get_dataframe(config)

    @staticmethod
    def _get_metabolite_names():
        """
        We need to extract metabolite names from the raw metabolites table
        """
        raw_metabolites = pd.read_csv(
            f'{DataLoader.DATA_FOLDER}metabolites_dataset.data_prep.tsv', delimiter='\t')

        return raw_metabolites['metabolite_id'].unique()
