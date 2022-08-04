import pandas as pd

from .dataloader import DataLoader


class TaskLoader:
    def get_simple(self, data: pd.DataFrame):
        """
        Orient the dataframe such that we have a list of features for each metabolites
        If measurements are available for multiple metabolites given a genotype stack measurements

        Parameters
        ----------
        data: pd.DataFrame
            Input dataframe that needs to be transformed to simple task format
        """
        not_meta_names = data.columns[~data.columns.isin(
            DataLoader._get_metabolite_names())].to_numpy()

        result = data.melt(id_vars=not_meta_names, var_name='metabolite_id',
                           value_name='metabolite_concentration', ignore_index=False)
        result = result.dropna(0)

        return result
