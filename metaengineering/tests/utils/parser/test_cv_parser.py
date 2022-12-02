import unittest
from unittest.mock import Mock, patch

import pandas as pd
from pandas.testing import assert_frame_equal

import numpy as np

from metaengineering.src.parsers.cv_parser import fmt_cv_results

class TestCVParser(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
    
    def test_cv_parser(self):
        dir = "/home/tvangraft/tudelft/thesis/metaengineering/data/results/experiment_0"
        df = pd.read_csv(f'{dir}/Strategy.ALL_all.csv')
        df = fmt_cv_results(df)
        self.assertTrue(True)