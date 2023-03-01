from src.pipeline.dataloader import DataLoader
from src.pipeline.taskloader import TaskLoader, TaskFrame
from src.pipeline.config import TaskLoaderConfig, DataLoaderConfig

from src.settings.tier import Tier
from src.settings.strategy import Strategy
from src.settings.model import HYPERPARAMETERS

from src.utils.utils import build_config, get_generator

import itertools

from sklearn.model_selection import GroupShuffleSplit 

import numpy as np
import pandas as pd

DataLoader.DATA_FOLDER = './data/training/'

def get_dl_config_for_strategy(tier: Tier):
    lookup = {
        Tier.TIER0: dict(
            additional_filters=["is_precursor",],
            additional_transforms=["log_fold_change_protein",]
        ),
        Tier.TIER1: dict(
            additional_frames=["interaction_frame",],
            additional_filters=["is_precursor"],
            additional_transforms=["log_fold_change_protein", "ppi_coo_matrix"]
        )
    }
    return lookup.get(tier)

def get_dataloader(dl_config):
    dl = DataLoader()
    dl.prepare_dataloader(dl_config)
    return dl
    
def get_taskloader(tl_config):
    tl = TaskLoader()
    tl.prepare_taskloader(tl_config)
    return tl

def train_test_split(
        tf: TaskFrame,
        strategy: Strategy,
        tier: Tier,
        test_size=0.3,
        shuffle=False,
        stratify=None,
    ):
    X_df, y_df = tf.x.reset_index(), tf.y.reset_index()

    if strategy == strategy.ONE_VS_ALL:
        metabolite_id = tf.frame_name
        X_train, X_test = X_df[X_df['metabolite_id'] != metabolite_id], X_df[X_df['metabolite_id'] == metabolite_id]
        y_train, y_test = tf.y[X_train.index], tf.y[X_test.index]
    elif strategy == strategy.ALL and tier == Tier.TIER0: 
        # We need to make sure that all of the knockout samples in the training dataset or in the testing dataset
        knockouts = X_df['KO_ORF']
        metabolite_id = X_df['metabolite_id']
        sample_id = np.arange(len(knockouts))
        gss = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=2)
        splitter = gss.split(sample_id, metabolite_id, knockouts)

        train_idx, test_idx = next(splitter)
        X_train, X_test, y_train, y_test = X_df.iloc[train_idx], X_df.iloc[test_idx], y_df.iloc[train_idx], y_df.iloc[test_idx]

        # Assert that there is a fair distrubition of metabolites
        base_distribution = X_df['metabolite_id'].value_counts().to_dict()

        print(base_distribution)
        print(X_test['metabolite_id'].value_counts().to_dict())

        actual_distribution = [np.isclose([value], [base_distribution[key] * test_size], atol=2)[0] for key, value in X_test['metabolite_id'].value_counts().to_dict().items()]
        assert(all(actual_distribution))

        # Assert that the knockouts are either in train or in test
        train_knockouts = X_train['KO_ORF'].isin(X_test['KO_ORF']).values
        test_knockouts = X_test['KO_ORF'].isin(X_train['KO_ORF']).values

        assert(not(any(train_knockouts)))
        assert(not(any(test_knockouts)))
    else:
        _x_train_df = pd.read_csv(f'./data/preprocessed/x_train_{Tier.TIER0}_all_{Strategy.ALL}.csv')
        _x_test_df = pd.read_csv(f'./data/preprocessed/x_test_{Tier.TIER0}_all_{Strategy.ALL}.csv')

        X_train = X_df[X_df['KO_ORF'].isin(_x_train_df['KO_ORF'])]
        print(X_train)
        X_test = X_df[X_df['KO_ORF'].isin(_x_test_df['KO_ORF'])]
        print(y_df)
        y_train = y_df.loc[X_train.index]
        y_test = y_df.loc[X_test.index]

        if 'metabolite_id' not in y_train.columns:
            y_train = y_train.assign(metabolite_id=tf.frame_name)
            y_test = y_test.assign(metabolite_id=tf.frame_name)

    X_train.to_csv(f'./data/preprocessed/x_train_{tier}_{tf.frame_name}_{strategy}.csv')
    X_test.to_csv(f'./data/preprocessed/x_test_{tier}_{tf.frame_name}_{strategy}.csv')
    y_train.to_csv(f'./data/preprocessed/y_train_{tier}_{tf.frame_name}_{strategy}.csv')
    y_test.to_csv(f'./data/preprocessed/y_test_{tier}_{tf.frame_name}_{strategy}.csv')

def main():
    tiers = [Tier.TIER0, Tier.TIER1]
    strategies = [Strategy.ALL, Strategy.METABOLITE_CENTRIC, Strategy.ONE_VS_ALL]

    for tier, strategy in itertools.product(tiers, strategies):
        print(f"{tier} - {strategy}")
        dl_config, tl_config, run_config, exp_config = build_config(
            strategy=strategy,
            tier=tier,
            params=HYPERPARAMETERS,
            forced_training=False,
            forced_testing=False,
            forced_shap=True,
            forced_lime=True,
            **get_dl_config_for_strategy(tier)
        )

        dl, tl = get_dataloader(dl_config), get_taskloader(tl_config)
        gen = get_generator(dl, tl, strategy, tier)
        for tf in gen:
            print(f"{tf.title=}")
            train_test_split(tf, strategy=strategy, tier=tier)
        

if __name__ == '__main__':
    main()