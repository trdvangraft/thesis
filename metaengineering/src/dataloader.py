import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from metaengineering.src.settings import DataOrientation


class DataLoader:
    DATA_FOLDER = './metaengineering/data/'

    def get_simple_protein_metabolite_dataframe(self):
        """
        This dataframe is the simplest way for predicting the metabolite concentrations
        Produce a dataframe with the genotype as key and columns concatenated (genotype X (proteins + metabolites))

        Averages the repeated experiments of the raw protein dataset
        """
        raw_metabolites = pd.read_csv(
            f'{DataLoader.DATA_FOLDER}metabolites_dataset.data_prep.tsv', delimiter='\t')
        raw_proteins = pd.read_csv(
            f'{DataLoader.DATA_FOLDER}proteins_dataset.data_prep.tsv', delimiter='\t')

        # group and average repeated measurments and map to wide table
        LHS = raw_proteins.groupby(by=['KO_ORF', 'ORF'])['value'].mean(
        ).to_frame().pivot_table(index='KO_ORF', columns='ORF', values='value')
        RHS = raw_metabolites.pivot_table(
            index='genotype', columns='metabolite_id', values='value')

        return LHS.reset_index().merge(RHS, left_on='KO_ORF', right_on='genotype').set_index('KO_ORF')

    def get_simple(self, data: pd.DataFrame):
        """
        Orient the dataframe such that we have a list of features for each metabolites
        If measurements are available for multiple metabolites given a genotype stack measurements

        Parameters
        ----------
        data: pd.DataFrame
            Input dataframe that needs to be transformed to simple task format
        """
        meta_names = self._get_metabolite_names
        not_meta_names = data.columns[~data.columns.isin(
            meta_names)].to_numpy()

        result = data.melt(id_vars=not_meta_names, var_name='metabolite_id',
                           value_name='metabolite_concentration', ignore_index=False)
        result = result.dropna(0)

        return result

    @property
    def _get_metabolite_names(self):
        """
        We need to extract metabolite names from the raw metabolites table
        """
        raw_metabolites = pd.read_csv(
            f'{DataLoader.DATA_FOLDER}metabolites_dataset.data_prep.tsv', delimiter='\t')

        return raw_metabolites['metabolite_id'].unique()
