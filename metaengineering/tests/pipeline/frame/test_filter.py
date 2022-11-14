import unittest

import numpy as np
import pandas as pd

from metaengineering.src.pipeline.frame.filter import FrameFilters
from metaengineering.src.pipeline.frame.cache import FrameCache

class FrameFiltersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame_cache = FrameCache()
        self.filter = FrameFilters(self.frame_cache)

class FrameFiltersInteractionTest(FrameFiltersTest):
    def setUp(self) -> None:
        super().setUp()
        
    def test_all_data_is_removed(self):
        adj_matrix = self._get_interaction()
        self._set_ppi(adj_matrix)

        self.filter.has_at_least_n_interaction()
        self.assertTupleEqual(self._get_ppi_shape(), (0, 0))

    def test_all_data_remains(self):
        adj_matrix = self._get_interaction()
        adj_matrix = adj_matrix + 1
        self._set_ppi(adj_matrix)

        self.filter.has_at_least_n_interaction()
        self.assertTupleEqual(self._get_ppi_shape(), (10, 10))
    
    def test_all_enzymes_interact_with_same_enzyme(self):
        adj_matrix = self._get_interaction()
        adj_matrix[:, 0] = 1
        self._set_ppi(adj_matrix)

        self.filter.has_at_least_n_interaction()
        self.assertTupleEqual(self._get_ppi_shape(), (0, 0))
    
    def test_one_enzymes_interact_with_all_enzyme(self):
        adj_matrix = self._get_interaction()
        adj_matrix[0, :] = 1
        self._set_ppi(adj_matrix)

        self.filter.has_at_least_n_interaction()
        self.assertTupleEqual(self._get_ppi_shape(), (1, 10))
    
    def test_two_enzymes_interact_with_all_enzymes(self):
        adj_matrix = self._get_interaction()
        adj_matrix[[0, 1], :] = 1
        self._set_ppi(adj_matrix)

        self.filter.has_at_least_n_interaction()
        self.assertTupleEqual(self._get_ppi_shape(), (2, 10))

    def test_three_enzymes_interact_with_some_enzymes(self):
        adj_matrix = self._get_interaction()
        adj_matrix[[0, 1], :8] = 1 # I am included
        adj_matrix[[2], :9] = 1 # I am also included
        adj_matrix[[3], :7] = 1
        self._set_ppi(adj_matrix)

        self.filter.has_at_least_n_interaction()
        self.assertTupleEqual(self._get_ppi_shape(), (3, 10))
    

    
    def _get_interaction(self):
        return np.matrix([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
    
    def _set_ppi(self, adj_matrix):
        labels = [f'enz_{i}' for i in range(10)]
        self.frame_cache.update_frame(
            'ppi', pd.DataFrame(adj_matrix, columns=labels, index=labels)
        )
    
    def _get_ppi_shape(self):
        return self.frame_cache.get_frame('ppi').shape