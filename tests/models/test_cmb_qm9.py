import torch

#configs
from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.networks.mixed_networks_config import MixedEGNN_dynamics_QM9Config

#models
from markov_bridges.models.generative_models.cmb_lightning import (
    MixedForwardMapL,
    MixedForwardMapMoleculesL
)

# loaders
from markov_bridges.utils.paralellism import check_model_devices
from markov_bridges.data.dataloaders_utils import get_dataloaders
from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network

def test_forward(config):
    dataloader = get_dataloaders(config)
    model = MixedForwardMapMoleculesL(config,dataloader)
    databatch = dataloader.get_databatch()
    databatch_nametuple = model.prepare_batch(databatch)

    # sample bridge
    discrete_sample, continuous_sample = model.sample_bridge(databatch_nametuple)

    # forward
    discrete_head,continuous_head = model.mixed_network(discrete_sample,
                                                        continuous_sample,
                                                        databatch_nametuple.time,
                                                        databatch_nametuple)    
    print(discrete_sample.shape)
    print(discrete_head.shape)

    print(continuous_sample.shape)
    print(continuous_head.shape)

def test_loss(config):
    dataloader = get_dataloaders(config)
    model = MixedForwardMapMoleculesL(config,dataloader)
    databatch = dataloader.get_databatch()
    databatch_nametuple = model.prepare_batch(databatch)
    # sample bridge
    discrete_sample, continuous_sample = model.sample_bridge(databatch_nametuple)
    # loss    
    full_loss,discrete_loss_,continuous_loss_ = model.loss(databatch_nametuple,discrete_sample,continuous_sample)
    print(full_loss)

def test_sample(config):
    dataloader = get_dataloaders(config)
    model = MixedForwardMapMoleculesL(config,dataloader)

    n_samples = 4
    device = check_model_devices(model.mixed_network)
    max_n_nodes,nodesxsample,node_mask,edge_mask,context = model.sample_sizes_and_masks(sample_size=n_samples,
                                                                                        device=device)
    t = torch.rand(size=(n_samples, 1),device=device)

    zt = model.sample_combined_position_feature_noise(n_samples, max_n_nodes, node_mask)
    print(zt.shape)
    eps_t = model.mixed_network.phi(t,zt, node_mask, edge_mask, context)
    
    print(eps_t.shape)
    return eps_t

if __name__=="__main__":
    config = CMBConfig()
    config.data = QM9Config(num_pts_train=1000,
                            num_pts_test=200,
                            num_pts_valid=200,
                            include_charges=False)
    config.mixed_network = MixedEGNN_dynamics_QM9Config(n_layers=1,
                                                        conditioning=['H_thermo', 'homo'])
    test_forward(config)



