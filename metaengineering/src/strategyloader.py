from typing import Tuple

from src.settings import Strategy

import pandas as pd
import numpy as np


class StrategyLoader:
    def get_dataset(self,
                    strategy: Strategy,
                    data: pd.DataFrame = None,
                    ):
        if strategy == Strategy.ALL:
            return self.get_all(data)
        elif strategy == Strategy.GENOTYPE_CENTRIC:
            return self.get_genotype_centric(data)
        elif strategy == Strategy.METABOLITE_CENTRIC:
            return self.get_metabolite_centric(data)
        elif strategy == Strategy.MODEL_CENTRIC:
            return self.get_model_centric(data)
        else:
            raise TypeError('Select a strategy that is defined')

    def get_all(self, data: pd.DataFrame):
        yield self._split_frame(data)

    def get_model_centric(self, data: pd.DataFrame):
        # exchange information between dataframes for efficient pandas group by
        data = data.reset_index()
        genotypes = data['KO_ORF'].unique()

        for genotype in genotypes:
            df: pd.DataFrame = data[data['KO_ORF'] == genotype]
            metabolites = df['metabolite_id'].unique()
            for metabolite in metabolites:
                yield self._split_frame(df[df['metabolite_id'] == metabolite])

    def get_genotype_centric(self, data: pd.DataFrame):
        data = data.reset_index()
        genotypes = data['KO_ORF'].unique()
        for genotype in genotypes:
            yield self._split_frame(data[data['KO_ORF'] == genotype])

    def get_metabolite_centric(self, data: pd.DataFrame):
        metabolites = data['metabolite_id'].unique()
        for metabolite in metabolites:
            yield self._split_frame(data[data['metabolite_id'] == metabolite])

    def _split_frame(self, df: pd.DataFrame):
        x: pd.DataFrame = df[df.columns[~df.columns.isin(
            ['metabolite_id', 'metabolite_concentration'])]]
        y = df[df.columns[df.columns.isin(
            ['metabolite_id', 'metabolite_concentration'])]]

        if 'KO_ORF' in x.columns:
            x: pd.DataFrame = x.set_index('KO_ORF')

        y: pd.DataFrame = y.set_index('metabolite_id')
        return x, y
