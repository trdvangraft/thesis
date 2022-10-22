from collections import defaultdict
from typing import List

import pandas as pd
import numpy as np

from sklearn.compose import TransformedTargetRegressor

from scipy.stats import pearsonr


def gather_results(paths: List[str]):
    metabolites_names = [path.rsplit('/', 1)[1].removesuffix('.csv').removeprefix('Strategy.METABOLITE_CENTRIC_') for path in paths]
    df = pd.concat([
        pd.read_csv(path).assign(metabolite_id=metabolite_name)
        for path, metabolite_name in zip(paths, metabolites_names)
    ])
    return df


class TestResultStore:
    def __init__(
        self,
        experiment_path: str,
        strategy: str,
    ) -> None:
        self.results = defaultdict(lambda: list())
        self.pred_results = defaultdict(lambda: dict())

        self.experiment_path = experiment_path
        self.strategy = strategy

    def update_results(
        self,
        key: str,
        model: TransformedTargetRegressor,
        architecture: str,
        X_test: np.array,
        y_test: np.array
    ):
        if len(X_test) < 2:
            return

        self.results[f"{key}_{architecture}"].append(pearsonr(y_test, model.predict(X_test))[0])
        self.pred_results[f"{key}_{architecture}"].update({
            'y_true': y_test.values,
            'y_pred': model.predict(X_test),
            'architecture': architecture,
            'metabolite_id': key,
            'correlation': pearsonr(y_test, model.predict(X_test))[0],
        })

    def to_file(self):
        pd.DataFrame.from_dict(self.results).to_csv(f'{self.experiment_path}/best_model_performance_{self.strategy}.csv')
        pd.DataFrame.from_dict(self.pred_results).to_json(f'{self.experiment_path}/best_model_prediction_performance_{self.strategy}.json')
