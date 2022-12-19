from typing import List

import torch

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.nn import Node2Vec

import pandas as pd
import numpy as np

from src.gnn.graph_builder import edge_index_from_df

def generate_embedding(
    edge_index: torch.Tensor,
    device,
    hetero=True,
):
    def get_mask(node_set, nodes):
        mask = torch.zeros(len(nodes), dtype=torch.long, device=device)
        for i in node_set:
            mask[np.argwhere(nodes == i)[0][0]] = 1.
        return mask

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(
                pos_rw.to(device), 
                neg_rw.to(device)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    if hetero:
        edge_index[:, 1] = edge_index[:, 1] + edge_index[:, 0].max()

    nodes = np.concatenate((np.unique(edge_index[:,0]), np.unique(edge_index[:,1])))
    num_nodes = len(nodes)

    np.random.shuffle(nodes) # shuffle node order

    train_size = int(num_nodes*0.7)
    test_size = int(num_nodes*0.85) - train_size
    val_size = num_nodes - train_size - test_size

    train_set = nodes[:train_size]
    test_set = nodes[train_size:train_size+test_size]
    val_set = nodes[train_size+test_size:]

    assert len(train_set) + len(test_set) + len(val_set) == num_nodes
    print("train set\t",train_set[:10])
    print("test set \t",test_set[:10])
    print("val set  \t",val_set[:10])

    train_mask = get_mask(train_set, nodes)
    test_mask = get_mask(test_set, nodes)
    val_mask = get_mask(val_set, nodes)

    print("train mask \t",train_mask[0:15])
    print("test mask  \t",test_mask[0:15])
    print("val mask   \t",val_mask[0:15])

    featureless_sample = Data(
        edge_index=edge_index.clone().detach().T,
        train_mask=train_mask,
        test_mask=test_mask,
        val_mask=val_mask,
    )

    model = Node2Vec(
        featureless_sample.edge_index,
        embedding_dim=32,
        walk_length=20,
        context_size=10,
        num_nodes=num_nodes,
    )

    loader = model.loader(batch_size=32, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 101):
        loss = train()
        #acc = test()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    node_embeddings = model()
    return node_embeddings