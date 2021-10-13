from typing import *

import json
import os
import tqdm
import networkx as nx
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

DATA_FOLDER = "../../data"


def load_data() -> Tuple[Generator, dict]:
    """Loads the data

    :return: social graph and tweet cascades
    :rtype: Tuple[Generator, dict]
    """
    graph_path = os.path.join(DATA_FOLDER, 'graph.gexf')
    G = nx.read_gexf(graph_path)

    cascades_path = os.path.join(DATA_FOLDER, 'cascades.json')

    with open(cascades_path, 'r') as f:
        cascades = json.load(f)

    return G, cascades

def prepare_data() -> Tuple[Data, List[Dict[str, torch.Tensor]]]:
    """First data preparation step for PyTorch

    :return: The graph and a list of samples (x, y), where
                x: node signal of early adopters
                y: node target signal of final adopters
    :rtype: Tuple[Data, List[Dict[str, torch.Tensor]]]
    """
    G, cascades = load_data()

    edge_index = torch.Tensor([[int(tup[0]), int(tup[1])] for tup in list(G.edges)]).long()
    edge_index = edge_index.transpose(0, 1).contiguous().view(2, -1)

    data = {'edge_index': edge_index}

    for node_id, feat_dict in G.nodes(data=True):
        for key, value in feat_dict.items():
            data[key] = data.get(key, []) + [value]

    graph = Data.from_dict(data)

    graph.num_nodes = G.number_of_nodes()

    samples = []
    num_nodes = G.number_of_nodes()
    for hashtag, cascade in tqdm(cascades.items()):
        early_adopt = cascade['early']
        final_adopt = cascade['final']

        x = torch.zeros(num_nodes)
        x[early_adopt] = 1
        x = x.unsqueeze(-1).float()

        y = torch.zeros(num_nodes)
        y[final_adopt] = 1
        y = y.long()

        sample = {'x': x, 'y': y}
        samples.append(sample)
    
    return graph, samples

class GraphDataset(Dataset):
    def __init__(self, graph, samples, target):
        """
            graph: the fixed graph encoding the underlying structure
            features: list (num_samples) containing tensors (num_nodes, feature_dim)
            labels: list (num_samples) containing tensors (num_nodes, feature_dim)
        """
        self.graph = graph 
        self.feature_names = self.get_feature_names()
        self.static_features = self.get_static_features()
        self.samples = samples
        self.target = target

    def __getitem__(self, idx):
        x = self.samples[idx]['x']
        y = self.samples[idx]['y']
        edge_index = self.graph.edge_index
        static_features = self.static_features
        
        # Popularity of hashtag
        if self.target == 'graph':
            y = torch.sum(y, dim=-1)
        elif self.target == 'node':
            pass
        else:
            raise NotImplementedError

        return {'x': x, 'y': y, 'edge_index': edge_index, 'static_features': static_features}

    def get_static_features(self):
        static_features = torch.stack([torch.tensor(self.graph[static_feature]) for static_feature in self.feature_names ], dim=1)
        return static_features

    def get_feature_names(self):
        features = self.graph.keys
        features.remove('label')
        features.remove('edge_index')
        return features

    def __len__(self):
        return len(self.samples)

