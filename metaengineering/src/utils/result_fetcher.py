from src.settings.tier import Tier
from src.settings.strategy import Strategy

import pandas as pd
import numpy as np

import os

class ResultFetcher:
    def __init__(self, root_dir: str, metabolite_info: pd.DataFrame) -> None:
        self.root_dir = root_dir
        self.metabolite_info = metabolite_info
    
    def get_all(self, experiment_id: Tier):
        test_df_all = self.get_test_df_all(experiment_id)
        test_df_metabolite = self.get_test_df_metabolite(experiment_id)
        test_df_one_vs_all = self.get_test_df_one_vs_all(experiment_id)
        return test_df_all, test_df_metabolite, test_df_one_vs_all
    
    def get_test_df_all(self, experiment_id: Tier):
        df = self.get_frame(experiment_id, Strategy.ALL)
        df = df.assign(strategy='all').assign(experiment_id=experiment_id)
        return df
    
    def get_test_df_metabolite(self, experiment_id: Tier):
        df = self.get_frame(experiment_id, Strategy.METABOLITE_CENTRIC)
        df = df.assign(strategy='metabolite').assign(experiment_id=experiment_id)
        return df
    
    def get_test_df_one_vs_all(self, experiment_id: Tier):
        df = self.get_frame(experiment_id, Strategy.ONE_VS_ALL)
        df = df.assign(strategy='one_vs_all').assign(experiment_id=experiment_id)
        return df
        
    def get_frame(self, experiment_id: Tier, strategy: Strategy):
        path = f"{self.root_dir}/{experiment_id}/best_model_performance_{strategy}.csv"
        print(path)
        if self.file_exists(path):
            test_df_all = pd.read_csv(path, index_col=0)
            test_df_all = test_df_all.stack().to_frame().reset_index(1).set_axis(['metabolite_arch', 'r2'], axis=1)
            test_df_all[['metabolite_id', 'architecture']] = test_df_all['metabolite_arch'].str.split("_", expand=True)
            test_df_all = test_df_all.drop('metabolite_arch', axis=1).merge(self.metabolite_info, left_on='metabolite_id', right_index=True)
            return test_df_all
    
    def file_exists(self, path: str):
        return os.path.exists(path)