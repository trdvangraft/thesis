import pandas as pd
import numpy as np

class DataLoader:
    DATA_FOLER = './data/'

    def get_simple_protein_metabolite_dataframe(self):
        """
        This dataframe is the simplest way for predicting the metabolite concentrations
        Produce a dataframe with the genotype as key and columns concatenated (genotype X (proteins + metabolites))

        Averages the repeated experiments of the raw protein dataset
        """
        raw_metabolites = pd.read_csv('./data/metabolites_dataset.data_prep.tsv', delimiter='\t')
        raw_proteins = pd.read_csv('./data/proteins_dataset.data_prep.tsv', delimiter='\t')

        # group and average repeated measurments and map to wide table
        LHS = raw_proteins.groupby(by=['KO_ORF', 'ORF'])['value'].mean().to_frame().pivot_table(index='KO_ORF', columns='ORF', values='value')
        RHS = raw_metabolites.pivot_table(index='genotype', columns='metabolite_id', values='value')

        return LHS.reset_index().merge(RHS, left_on='genotype', right_on='KO_ORF').set_index('genotype')

