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

from torch_geometric.data import HeteroData
from torch_geometric.utils import from_networkx, to_networkx
import torch_geometric.transforms as T

from src.gnn.data_augmentation import DataAugmentation
from src.settings.metabolites import ENZYMES

def edge_index_from_df(
    graph_fc_df : pd.DataFrame, 
    edge_list: pd.DataFrame,
    valid_metabolites: List[str],
):
    data_augmentation = DataAugmentation(valid_metabolites, ENZYMES, graph_fc_df)

    first_row = graph_fc_df.iloc[[0]]
    enzyme_nodes = np.array([str(key) for key in first_row.to_dict().keys() if data_augmentation.data_augmentation_is_enzyme(key)])
    metabolite_nodes = np.array([str(key) for key in first_row.to_dict().keys() if data_augmentation.data_augmentation_is_metabolite(key)])

    edges = [
        [np.where(enzyme_nodes == enzyme)[0][0], np.where(metabolite_nodes == metabolite)[0][0]]
        for enzyme, metabolite in zip(edge_list['enzyme'], edge_list['metabolite_id'])
    ]

    return torch.Tensor(edges).to(torch.long).T

def from_node_attributes(attributes: Dict[Hashable, dict[str, Any]]):
    def collect_from_key(data, key: str):
        return [node[key] for id, node in data.items()]

    node_attributes = list(attributes[list(attributes.keys())[0]].keys())
    return torch.Tensor([collect_from_key(attributes, attribute) for attribute in node_attributes]).T.contiguous()

def get_samples_hetero_graph(
    target_metabolite_id: str,
    valid_metabolites: List[str],
    graph_fc_df: pd.DataFrame,
    edge_list_df: pd.DataFrame,
    node_embeddings: np.ndarray,
):
    samples: List[HeteroData] = []
    data_augmentation = DataAugmentation(
        valid_metabolites, ENZYMES, graph_fc_df
    )

    edge_index = edge_index_from_df(graph_fc_df, edge_list_df)

    for idx, row_series in graph_fc_df.iterrows():
        # print(row_series.to_dict())
        enzyme_attributes = {
            key: { 
                "fc": value, 
            } 
            for key, value in row_series.to_dict().items()
            if data_augmentation.data_augmentation_is_enzyme(key)
        }

        metabolite_attributes = {
            key: {
                "fc": value,
                "train_mask": data_augmentation.data_augmentation_train_mask(key, target_metabolite_id, value),
                "test_mask": data_augmentation.data_augmentation_test_mask(key, target_metabolite_id), 
                # "pathway": data_augmentation_get_pathway(key),
            } 
            for key, value in row_series.to_dict().items()
            if data_augmentation.data_augmentation_is_metabolite(key)
        }

        # print(from_node_attributes(enzyme_attributes).shape)
        # print(from_node_attributes(metabolite_attributes).shape)

        enzyme_features = from_node_attributes(enzyme_attributes)
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
    return samples

def get_graph_fc(
    edge_list_df, 
    valid_metabolites
):
    DataLoader.DATA_FOLDER = f'{get_project_root()}/data/training/'

    tier = Tier.TIER0
    strategy = Strategy.ALL

    dl_config = DataLoaderConfig(
        additional_filters=["is_precursor", ],
        additional_transforms=["log_fold_change_protein", ]
    )

    tl_config = TaskLoaderConfig(
        data_throttle=1,
        tier=tier,
    )

    dl = DataLoader()
    dl.prepare_dataloader(dl_config)

    tl = TaskLoader()
    tl.prepare_taskloader(tl_config)

    gen = get_generator(dl, tl, strategy, tier)
    tf = next(gen)
    trainer = Trainer()

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    df_x = tf.x.reset_index(level="metabolite_id", drop=True).drop_duplicates()
    df_y: pd.DataFrame = tf.y.to_frame().reset_index().pivot_table(values='metabolite_concentration', index='KO_ORF', columns='metabolite_id').loc[df_x.index]

    scaled_x = x_scaler.fit_transform(df_x)
    scaled_y = pd.DataFrame(y_scaler.fit_transform(df_y), columns=df_y.columns, index=df_y.index)

    matching_precursor_metabolite = [
        (metabolite_id, scaled_y.loc[:, metabolite_id].values) 
        for cobra_metabolite_id in edge_list_df['metabolite_id'].unique()
        if (metabolite_id := list(filter(lambda p: p in cobra_metabolite_id, valid_metabolites))[0])
    ]

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