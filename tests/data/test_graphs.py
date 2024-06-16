import pytest

from markov_bridges.configs.config_classes.data.graphs_configs import GraphDataloaderGeometricConfig,CommunitySmallGConfig
from markov_bridges.data.graphs_dataloader import GraphDataloader

def test_load_graph():
    data_config = CommunitySmallGConfig()
    dataloader = GraphDataloader(data_config)
    databatch = dataloader.get_databatch()

    print(databatch.source_discrete.shape)
    print(databatch.target_discrete.shape)

if __name__=="__main__":
    test_load_graph()
    