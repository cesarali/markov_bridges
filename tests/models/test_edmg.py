import torch
from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config

from markov_bridges.configs.config_classes.generative_models.edmg_config import (
    EDMGConfig,
    NoisingModelConfig
)
from markov_bridges.models.generative_models.edmg_lightning import EDGML
from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.utils.paralellism import check_model_devices


def test_forward_pass(config):
    edmg = EDGML(config)
    n_samples = 4
    device = check_model_devices(edmg.model.noising_model)
    max_n_nodes,nodesxsample,node_mask,edge_mask,context = edmg.model.sample_sizes_and_masks(sample_size=n_samples,device=device)
    t = torch.rand(size=(n_samples, 1),device=device)

    #one_hot, charges, x,node_mask = edmg.pipeline.sample(sample_size=10)
    zt = edmg.model.noising_model.sample_combined_position_feature_noise(n_samples, max_n_nodes, node_mask)
    eps_t = edmg.model.noising_model.phi(zt, t, node_mask, edge_mask, context)
    print(eps_t.shape)


if __name__=="__main__":
    config = EDMGConfig()
    config.data = QM9Config(num_pts_train=1000,
                            num_pts_test=200,
                            num_pts_valid=200)    
    config.noising_model = NoisingModelConfig(n_layers=2,
                                              conditioning=['H_thermo', 'homo'])
    # conditioning=['H_thermo', 'homo']
    config.trainer.metrics = []
    experiment_files = ExperimentFiles(experiment_name="cjb",
                                       experiment_type="graph",
                                       experiment_indentifier="lightning_test9",
                                       delete=True)
    test_forward_pass(config)

    