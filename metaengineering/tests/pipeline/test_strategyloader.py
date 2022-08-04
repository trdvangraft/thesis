import unittest

from src.pipeline.dataloader import DataLoader
from src.pipeline.taskloader import TaskLoader
from src.pipeline.strategyloader import StrategyLoader

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal


class TestStrategyLoader(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataloader = DataLoader()
        self.taskloader = TaskLoader()
        self.strategyLoader = StrategyLoader()

    def test_get_simple(self):
        x, y = next(self.strategyLoader.get_all(self._get_simple_dataset()))

        expected_x = pd.DataFrame(data={
            'GEN1': [0, 5, 0, 5],
            'GEN2': [1, 6, 1, 6],
            'GEN3': [2, 7, 2, 7],
            'KO_ORF': ['GENOTYPE_1', 'GENOTYPE_2', 'GENOTYPE_1', 'GENOTYPE_2']
        }).set_index('KO_ORF')

        expected_y = pd.DataFrame(data={
            'metabolite_id': ['atp', 'atp', 'adp', 'adp'],
            'metabolite_concentration': [3, 8, 4, 9],
        }).set_index('metabolite_id')

        assert_frame_equal(x, expected_x)
        assert_frame_equal(y, expected_y)

    def test_get_metabolite_centric(self):
        generator = self.strategyLoader.get_metabolite_centric(
            self._get_simple_dataset())

        with self.subTest('ATP metabolites'):
            x, y = next(generator)

            expected_x = pd.DataFrame(data={
                'GEN1': [0, 5],
                'GEN2': [1, 6],
                'GEN3': [2, 7],
                'KO_ORF': ['GENOTYPE_1', 'GENOTYPE_2']
            }).set_index('KO_ORF')

            expected_y = pd.DataFrame(data={
                'metabolite_id': ['atp', 'atp'],
                'metabolite_concentration': [3, 8],
            }).set_index('metabolite_id')

            assert_frame_equal(x, expected_x)
            assert_frame_equal(y, expected_y)

        with self.subTest('ADP metabolites'):
            x, y = next(generator)

            expected_x = pd.DataFrame(data={
                'GEN1': [0, 5],
                'GEN2': [1, 6],
                'GEN3': [2, 7],
                'KO_ORF': ['GENOTYPE_1', 'GENOTYPE_2'],
            }).set_index('KO_ORF')

            expected_y = pd.DataFrame(data={
                'metabolite_id': ['adp', 'adp'],
                'metabolite_concentration': [4, 9],
            }).set_index('metabolite_id')

            assert_frame_equal(x, expected_x)
            assert_frame_equal(y, expected_y)

    def test_get_genotype_centric(self):
        generator = self.strategyLoader.get_genotype_centric(
            self._get_simple_dataset())

        with self.subTest('Genotype 1'):
            x, y = next(generator)

            expected_x = pd.DataFrame(data={
                'GEN1': [0, 0],
                'GEN2': [1, 1],
                'GEN3': [2, 2],
                'KO_ORF': ['GENOTYPE_1', 'GENOTYPE_1'],
            }).set_index('KO_ORF')

            expected_y = pd.DataFrame(data={
                'metabolite_id': ['atp', 'adp'],
                'metabolite_concentration': [3, 4],
            }).set_index('metabolite_id')

            assert_frame_equal(x, expected_x)
            assert_frame_equal(y, expected_y)

        with self.subTest('Genotype 2'):
            x, y = next(generator)

            expected_x = pd.DataFrame(data={
                'GEN1': [5, 5],
                'GEN2': [6, 6],
                'GEN3': [7, 7],
                'KO_ORF': ['GENOTYPE_2', 'GENOTYPE_2']
            }).set_index('KO_ORF')

            expected_y = pd.DataFrame(data={
                'metabolite_id': ['atp', 'adp'],
                'metabolite_concentration': [8, 9],
            }).set_index('metabolite_id')

            assert_frame_equal(x, expected_x)
            assert_frame_equal(y, expected_y)

    def test_get_model_centric(self):
        with self.subTest('Nine combinations can be made'):
            generator = self.strategyLoader.get_model_centric(
                self._get_complex_dataset())
            self.assertEqual(9, sum(1 for _ in generator))

        generator = self.strategyLoader.get_model_centric(
            self._get_complex_dataset())

        metabolites = ['atp', 'adp', 'amp']
        genotypes = ['GENOTYPE_1', 'GENOTYPE_2', 'GENOTYPE_3']

        # genotype loop
        for i in range(3):
            # metabolite loop
            for j in range(3):
                with self.subTest(f'{genotypes[i]} with {metabolites[j]}, ({i}, {j})'):
                    x, y = next(generator)

                    expected_x = pd.DataFrame(data={
                        'GEN1': [0 + 5 * i],
                        'GEN2': [1 + 5 * i],
                        'GEN3': [2 + 5 * i],
                        'KO_ORF': [genotypes[i]]
                    }).set_index('KO_ORF')

                    expected_y = pd.DataFrame(data={
                        'metabolite_id': [metabolites[j]],
                        'metabolite_concentration': [3 + j + 5 * i],
                    }).set_index('metabolite_id')

                    assert_frame_equal(x, expected_x)
                    assert_frame_equal(y, expected_y)

    def test_happy_flow(self):
        df = self.dataloader.get_simple_protein_metabolite_dataframe()
        df = self.taskloader.get_simple(df)

        with self.subTest('get all'):
            x, y = next(self.strategyLoader.get_all(df))

            assert_index_equal(df.index, x.index)
            assert_series_equal(df['metabolite_id'],
                                y.index.to_series(), check_index=False)

        with self.subTest('Genotype centric'):
            x, y = next(self.strategyLoader.get_genotype_centric(df))

            self.assertEqual(len(x.index.unique()), 1, "In a genotype focussed mode we want one genotype per time")
            self.assertEqual(726, x.shape[1], "All genes should be represented from get simmple")
            self.assertEqual(x.shape[0], y.shape[0], "X and Y should have the same length")

        with self.subTest('Metabolite centric'):
            x, y = next(self.strategyLoader.get_metabolite_centric(df))

            self.assertEqual(len(y.index.unique()), 1, "In a metabolite focussed mode we want one metabolite per time")
            self.assertEqual(726, x.shape[1], "All genes should be represented from get simmple")
            self.assertEqual(x.shape[0], y.shape[0], "X and Y should have the same length")

        with self.subTest('Model centric'):
            x, y = next(self.strategyLoader.get_model_centric(df))

            self.assertEqual(len(x.index.unique()), 1, "In a model focussed mode we want one genotype per time")
            self.assertEqual(len(y.index.unique()), 1, "In a model focussed mode we want one metabolite per time")
            self.assertEqual(726, x.shape[1], "All genes should be represented from get simmple")
            self.assertEqual(x.shape[0], y.shape[0], "X and Y should have the same length")

    def _get_simple_dataset(self):
        return pd.DataFrame(data={
            'GEN1': [0, 5, 0, 5],
            'GEN2': [1, 6, 1, 6],
            'GEN3': [2, 7, 2, 7],
            'metabolite_id': ['atp', 'atp', 'adp', 'adp'],
            'metabolite_concentration': [3, 8, 4, 9],
            'KO_ORF': ['GENOTYPE_1', 'GENOTYPE_2', 'GENOTYPE_1', 'GENOTYPE_2']
        }).set_index('KO_ORF')

    def _get_complex_dataset(self):
        return pd.DataFrame(data={
            'GEN1': [0, 5, 10, 0, 5, 10, 0, 5, 10],
            'GEN2': [1, 6, 11, 1, 6, 11, 1, 6, 11],
            'GEN3': [2, 7, 12, 2, 7, 12, 2, 7, 12],
            'metabolite_id': ['atp', 'atp', 'atp', 'adp', 'adp', 'adp', 'amp', 'amp', 'amp'],
            'metabolite_concentration': [3, 8, 13, 4, 9, 14, 5, 10, 15],
            'KO_ORF': ['GENOTYPE_1', 'GENOTYPE_2', 'GENOTYPE_3', 'GENOTYPE_1', 'GENOTYPE_2', 'GENOTYPE_3', 'GENOTYPE_1', 'GENOTYPE_2', 'GENOTYPE_3']
        }).set_index('KO_ORF')
