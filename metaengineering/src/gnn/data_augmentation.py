from dataclasses import dataclass
from typing import List

import pandas as pd

@dataclass
class DataAugmentation:
    valid_metabolites: List[str]
    valid_enzymes: List[str]
    graph_fc: pd.DataFrame

    def data_augmentation_is_metabolite(self, node_name: str):
        return node_name not in self.valid_enzymes

    def data_augmentation_is_enzyme(self, node_name: str):
        return node_name in self.valid_enzymes

    def data_augmentation_train_mask(self, node_name: str, target_metabolite_id: str, concentration: float):
        return node_name not in self.valid_enzymes and self._get_matching_precursor_metabolite(node_name) != target_metabolite_id and concentration != 0.0

    def data_augmentation_test_mask(self, node_name: str, target_metabolite_id: str):
        return node_name not in self.valid_enzymes and self._get_matching_precursor_metabolite(node_name) == target_metabolite_id

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

