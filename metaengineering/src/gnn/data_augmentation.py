from dataclasses import dataclass
from typing import List

import pandas as pd
from src.settings.strategy import Strategy
from src.settings.metabolites import PRECURSOR_METABOLITES

@dataclass
class DataAugmentation:
    valid_metabolites: List[str]
    valid_enzymes: List[str]
    graph_fc: pd.DataFrame

    def data_augmentation_is_metabolite(self, node_name: str):
        return node_name not in self.valid_enzymes

    def data_augmentation_is_enzyme(self, node_name: str):
        return node_name in self.valid_enzymes

    def data_augmentation_train_mask(self, 
        node_name: str, 
        target_metabolite_id: str, 
        concentration: float,
        knockout_id: str,
        X_train_df: pd.DataFrame,
        strategy: Strategy,
    ):
        if strategy == Strategy.ALL:
            return self._is_valid_metabolite(node_name)
        elif strategy == Strategy.METABOLITE_CENTRIC:
            return self._is_valid_metabolite(node_name) and self._is_target_metabolite(node_name, target_metabolite_id, knockout_id, X_train_df) and concentration != 0.0
        elif strategy == Strategy.ONE_VS_ALL:
            return self._is_valid_metabolite(node_name) and not self._is_metabolite_target(node_name, target_metabolite_id)

    def data_augmentation_test_mask(self, 
        node_name: str, 
        target_metabolite_id: str,
        concentration: float,
        knockout_id: str,
        X_test_df: pd.DataFrame
    ):
        return self._is_valid_metabolite(node_name) and self._is_target_metabolite(node_name, target_metabolite_id, knockout_id, X_test_df) and concentration != 0.0

    def data_augmentation_get_knockout_idx(self, knockout: str):
        return self.graph_fc.index.get_loc(knockout)

    def data_augmentation_get_knockout_label(self, knockout_id: int):
        return self.graph_fc.iloc[[knockout_id]].index

    def _get_matching_precursor_metabolite(
        self, 
        cobra_metabolite_id: str,
    ):
        for precursor_metabolite in self.valid_metabolites:
            if precursor_metabolite in cobra_metabolite_id:
                return precursor_metabolite
    
    def _is_valid_metabolite(self, node_name):
        return node_name not in self.valid_enzymes and node_name in PRECURSOR_METABOLITES

    def _is_metabolite_target(self, node_name, target_metabolite):
        matching_metabolite_name = self._get_matching_precursor_metabolite(node_name)
        return matching_metabolite_name == target_metabolite

    def _is_target_metabolite(self, node_name, target_metabolite, knockout_id, X_df):
        matching_metabolite_name = self._get_matching_precursor_metabolite(node_name)
        check_metabolite_in_df: bool = matching_metabolite_name in X_df['metabolite_id'].values
        check_knockout_in_df: bool = knockout_id in X_df['KO_ORF'].values
        check_metabolite_is_target: bool = self._is_metabolite_target(node_name, target_metabolite)
        check_row_exists: bool = ((X_df['metabolite_id'] == matching_metabolite_name) & (X_df['KO_ORF'] == knockout_id)).any()
        return check_metabolite_in_df and check_knockout_in_df and check_metabolite_is_target and check_row_exists

    # def _is_target_metabolite(self, node_name, target_metabolite_id):
    #     return self._get_matching_precursor_metabolite(node_name) == target_metabolite_id

