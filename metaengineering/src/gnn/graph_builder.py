from typing import DefaultDict, List, Hashable, Dict, Any

from src.utils.utils import get_generator, get_project_root

from src.pipeline.config import DataLoaderConfig, TaskLoaderConfig
from src.pipeline.taskloader import TaskLoader
from src.pipeline.dataloader import DataLoader

from src.orchestrator.trainer import Trainer

from src.settings.tier import Tier
from src.settings.strategy import Strategy

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import torch

from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import from_networkx, to_networkx
import torch_geometric.transforms as T

from src.gnn.data_augmentation import DataAugmentation
from src.settings.metabolites import ENZYMES, PRECURSOR_METABOLITES

def edge_index_from_df(
    graph_fc_df : pd.DataFrame, 
    edge_list: pd.DataFrame,
    valid_metabolites: List[str],
):
    data_augmentation = DataAugmentation(valid_metabolites, ENZYMES, graph_fc_df)

    # Get the names of the nodes that live in the graph
    first_row = graph_fc_df.iloc[[0]]
    enzyme_nodes = np.array([str(key) for key in first_row.to_dict().keys() if data_augmentation.data_augmentation_is_enzyme(key)])
    metabolite_nodes = np.array([str(key) for key in first_row.to_dict().keys() if data_augmentation.data_augmentation_is_metabolite(key)])

    edges = [
        [np.where(enzyme_nodes == enzyme)[0][0], np.where(metabolite_nodes == metabolite)[0][0]]
        for enzyme, metabolite in zip(edge_list['enzyme'], edge_list['metabolite_id'])
    ]

    return torch.Tensor(edges).to(torch.long).T

def edge_index_from_df_protein_only(
    graph_fc_df : pd.DataFrame, 
    edge_list: pd.DataFrame,
    valid_metabolites: List[str],
):
    data_augmentation = DataAugmentation(valid_metabolites, ENZYMES, graph_fc_df)

    # Get the names of the nodes that live in the graph
    first_row = graph_fc_df.iloc[[0]]
    enzyme_nodes = np.array([str(key) for key in first_row.to_dict().keys() if data_augmentation.data_augmentation_is_enzyme(key)])

    edges = [
        [np.where(enzyme_nodes == source_node)[0][0], np.where(enzyme_nodes == target_node)[0][0]]
        for source_node, target_node in zip(edge_list['source'], edge_list['target'])
    ]

    return torch.Tensor(edges).to(torch.long).T

def from_node_attributes(attributes: Dict[Hashable, Dict[str, Any]]):
    def collect_from_key(data, key: str):
        return [node[key] for id, node in data.items()]

    node_attributes = list(attributes[list(attributes.keys())[0]].keys())
    return torch.Tensor([collect_from_key(attributes, attribute) for attribute in node_attributes]).T.contiguous()

def get_enzyme_features(row_series, data_augmentation):
    enzyme_attributes = {
        key: { 
            "fc": value, 
        } 
        for key, value in row_series.to_dict().items()
        if data_augmentation.data_augmentation_is_enzyme(key)
    }
    return from_node_attributes(enzyme_attributes)

def get_samples_hetero_graph(
    target_metabolite_id: str,
    strategy: Strategy,
    valid_metabolites: List[str],
    graph_fc_df: pd.DataFrame,
    edge_list_df: pd.DataFrame,
    node_embeddings: np.ndarray,
):
    samples: List[HeteroData] = []

    train_samples: List[HeteroData] = []
    test_samples: List[HeteroData] = []

    data_augmentation = DataAugmentation(
        valid_metabolites, ENZYMES, graph_fc_df
    )

    X_train_df, X_test_df = get_knockout_orf(strategy, target_metabolite_id)

    edge_index = edge_index_from_df(graph_fc_df, edge_list_df, valid_metabolites)

    for knockout_id, row_series in graph_fc_df.iterrows():
        metabolite_attributes = {
            key: {
                "fc": value,
                "train_mask": data_augmentation.data_augmentation_train_mask(key, target_metabolite_id, value, knockout_id, X_train_df, strategy),
                "test_mask": data_augmentation.data_augmentation_test_mask(key, target_metabolite_id, value, knockout_id, X_test_df),
                # "pathway": data_augmentation_get_pathway(key),
            } 
            for key, value in row_series.to_dict().items()
            if data_augmentation.data_augmentation_is_metabolite(key)
        }

        # print(from_node_attributes(enzyme_attributes).shape)
        # print(from_node_attributes(metabolite_attributes).shape)

        enzyme_features = get_enzyme_features(row_series, data_augmentation)
        metabolite_features = from_node_attributes(metabolite_attributes)

        num_metabolite_nodes = metabolite_features.shape[0]
        num_enzyme_nodes = enzyme_features.shape[0]

        data = HeteroData()
        data['enzymes'].x = torch.hstack((enzyme_features, node_embeddings[:num_enzyme_nodes]))
        data["enzymes"].num_nodes = num_enzyme_nodes

        data['metabolites'].x = torch.hstack((metabolite_features[:, 3:], node_embeddings[num_enzyme_nodes:]))
        data['metabolites'].y = metabolite_features[:, 0]
        data["metabolites"].train_mask = metabolite_features[:, 1]
        data["metabolites"].test_mask = metabolite_features[:, 2]
        data["metabolites"].num_nodes = num_metabolite_nodes

        data['enzymes', 'regulates_reaction', 'metabolites'].edge_index = edge_index

        data = T.ToUndirected()(data)

        samples.append(data)

    for sample in samples:
        if strategy == Strategy.ONE_VS_ALL:
            train_samples.append(sample)
            test_samples.append(sample)
        elif sample['metabolites']['test_mask'].sum() > 0:
            test_samples.append(sample)
        elif sample['metabolites']['train_mask'].sum() > 0:
            train_samples.append(sample)

    return train_samples, test_samples

def get_samples_graph(
    target_metabolite_id: str,
    strategy: Strategy,
    valid_metabolites: List[str],
    graph_fc_df: pd.DataFrame,
    edge_index: torch.Tensor,
    node_embeddings: np.ndarray,
):
    samples: List[Data] = []

    train_samples: List[Data] = []
    test_samples: List[Data] = []

    data_augmentation = DataAugmentation(
        valid_metabolites, ENZYMES, graph_fc_df
    )

    X_train_df, X_test_df = get_knockout_orf(strategy, target_metabolite_id)

    for knockout_id, row_series in graph_fc_df.iterrows():
        metabolite_attributes = {
            key: {
                "fc": value,
                "train_mask": data_augmentation.data_augmentation_train_mask(key, target_metabolite_id, value, knockout_id, X_train_df, strategy),
                "test_mask": data_augmentation.data_augmentation_test_mask(key, target_metabolite_id, value, knockout_id, X_test_df),
                # "pathway": data_augmentation_get_pathway(key),
            } 
            for key, value in row_series.to_dict().items()
            if data_augmentation.data_augmentation_is_metabolite(key)
        }

        enzyme_features = get_enzyme_features(row_series, data_augmentation)
        metabolite_features = from_node_attributes(metabolite_attributes)

        num_enzyme_nodes = enzyme_features.shape[0]

        data = Data(
            x=torch.hstack((enzyme_features, node_embeddings[:num_enzyme_nodes])),
            edge_index=edge_index,
            y=metabolite_features[:, 0],
            num_nodes = num_enzyme_nodes,
            train_mask = metabolite_features[:, 1],
            test_mask = metabolite_features[:, 2],
        )

        samples.append(data)

    for sample in samples:
        if strategy == Strategy.ONE_VS_ALL:
            train_samples.append(sample)
            test_samples.append(sample)
        elif sample['test_mask'].sum() > 0:
            test_samples.append(sample)
        elif sample['train_mask'].sum() > 0:
            train_samples.append(sample)

    return train_samples, test_samples

def get_knockout_orf(strategy: Strategy, target_metabolite_id: str):
    tf = get_tf(strategy)
    trainer = Trainer()

    if strategy == Strategy.ALL:
        split_kwargs = dict(stratify='metabolite_id', shuffle=True)
    else:
        split_kwargs = dict(stratify=None, shuffle=False)

    train_dfs = []
    test_dfs = []

    for tf in get_tf(strategy):
        X_train, X_test, _, _ = trainer.do_train_test_split(tf, strategy, **split_kwargs)

        if 'metabolite_id' not in X_test.columns:
            X_test = X_test.assign(metabolite_id=tf.frame_name)
            X_train = X_train.assign(metabolite_id=tf.frame_name)

        train_dfs.append(X_train)
        test_dfs.append(X_test)
    
    X_train = pd.concat(train_dfs, axis=0)
    X_test = pd.concat(test_dfs, axis=0)

    
    return X_train[['KO_ORF', 'metabolite_id']], X_test[['KO_ORF', 'metabolite_id']]

def get_tf(strategy: Strategy):
    DataLoader.DATA_FOLDER = f'{get_project_root()}/data/training/'
    tier = Tier.TIER0

    dl_config = DataLoaderConfig(
        additional_filters=['is_precursor'],
        additional_transforms=["log_fold_change_protein"]
    )

    dl = DataLoader()
    dl.prepare_dataloader(dl_config)

    tl = TaskLoader()
    tl.prepare_taskloader(TaskLoaderConfig())

    gen = get_generator(dl, tl, strategy, tier)
    return gen

def get_graph_fc(
    edge_list_df, 
    valid_metabolites
):
    """
    Functions that transforms a taskframe to a dataframe containing the enzyme and metabolite fold changes
    """
    tf = next(get_tf(Strategy.ALL))

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    df_x = tf.x.reset_index(level="metabolite_id", drop=True).drop_duplicates()
    df_y: pd.DataFrame = tf.y.to_frame().reset_index().pivot_table(values='metabolite_concentration', index='KO_ORF', columns='metabolite_id').loc[df_x.index]

    scaled_x = x_scaler.fit_transform(df_x)
    scaled_y = pd.DataFrame(y_scaler.fit_transform(df_y), columns=df_y.columns, index=df_y.index)

    _metabolites_of_interest = list(set(valid_metabolites) & set(PRECURSOR_METABOLITES))
    print(f"{_metabolites_of_interest=}")

    matching_precursor_metabolite = []

    for cobra_metabolite_id in edge_list_df['metabolite_id'].unique():
        metabolite_id_list = list(filter(lambda p: p in cobra_metabolite_id, _metabolites_of_interest))

        if len(metabolite_id_list) > 0:
            metabolite_id_str = metabolite_id_list[0]
            matching_precursor_metabolite.append((metabolite_id_str, scaled_y.loc[:, metabolite_id_str].values))
        else:
            matching_precursor_metabolite.append((cobra_metabolite_id, np.zeros(shape=scaled_y.shape[0])))

    cobra_df_y = pd.DataFrame.from_records(matching_precursor_metabolite).T
    cobra_df_y = cobra_df_y.explode(cobra_df_y.columns.to_list()) \
        .drop(0) \
        .set_index(scaled_y.index) \
        .set_axis(edge_list_df['metabolite_id'].unique(), axis=1) \
        .fillna(0)
    cobra_df_y

    graph_fc_df = pd.concat([
        pd.DataFrame(scaled_x, index=df_x.index, columns=df_x.columns).loc[:, edge_list_df['enzyme'].unique()],
        cobra_df_y,
    ], axis=1).fillna(0)
    return graph_fc_df

def get_graph_fc_protein_only(
    edge_list_df, 
    valid_metabolites
):
    """
    Functions that transforms a taskframe to a dataframe containing the enzyme and metabolite fold changes
    """
    tf = next(get_tf(Strategy.ALL))

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    df_x = tf.x.reset_index(level="metabolite_id", drop=True).drop_duplicates()
    df_y: pd.DataFrame = tf.y.to_frame().reset_index().pivot_table(values='metabolite_concentration', index='KO_ORF', columns='metabolite_id').loc[df_x.index]

    scaled_x = x_scaler.fit_transform(df_x)
    scaled_y = pd.DataFrame(y_scaler.fit_transform(df_y), columns=df_y.columns, index=df_y.index)

    _metabolites_of_interest = list(set(valid_metabolites) & set(PRECURSOR_METABOLITES))

    graph_fc_df = pd.concat([
        pd.DataFrame(scaled_x, index=df_x.index, columns=df_x.columns).loc[:, pd.unique(edge_list_df[['source', 'target']].values.ravel('K'))],
        scaled_y.loc[:, _metabolites_of_interest],
    ], axis=1).fillna(0)
    return graph_fc_df