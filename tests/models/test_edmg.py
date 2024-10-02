import torch
from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config

from markov_bridges.configs.config_classes.generative_models.edmg_config import (
    EDMGConfig,
    NoisingModelConfig
)
from markov_bridges.models.generative_models.edmg_lightning import EquivariantDiffussionNoisingL,EDGML
from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.utils.paralellism import check_model_devices
from markov_bridges.data.qm9.utils import prepare_context


from markov_bridges.utils.equivariant_diffusion import (
    assert_mean_zero_with_mask, 
    remove_mean_with_mask,
    assert_correctly_masked, 
    sample_center_gravity_zero_gaussian_with_mask,
    random_rotation,
    Queue
)

from markov_bridges.data.dataloaders_utils import get_dataloaders

def test_forward_pass(config:EDMGConfig):
    dataloader = get_dataloaders(config)
    model = EquivariantDiffussionNoisingL(config,dataloader)
    n_samples = 4
    device = check_model_devices(model.noising_model)
    max_n_nodes,nodesxsample,node_mask,edge_mask,context = model.sample_sizes_and_masks(sample_size=n_samples,device=device)
    t = torch.rand(size=(n_samples, 1),device=device)

    #one_hot,charges,x,node_mask = edmg.pipeline.sample(sample_size=10)
    zt = model.noising_model.sample_combined_position_feature_noise(n_samples, max_n_nodes, node_mask)
    eps_t = model.noising_model.phi(zt, t, node_mask, edge_mask, context)
    print(eps_t.shape)

def test_loss(config:EDMGConfig):
    data_config = config.data
    dtype = torch.float32
    dataloader = get_dataloaders(config)
    model = EquivariantDiffussionNoisingL(config,dataloader)
    databatch = dataloader.get_databatch()
    
    x = databatch['positions'].to(dtype)
    node_mask = databatch['atom_mask'].to(dtype).unsqueeze(2)
    edge_mask = databatch['edge_mask'].to(dtype)
    one_hot = databatch['one_hot'].to(dtype)
    charges = (databatch['charges'] if data_config.include_charges else torch.zeros(0)).to(x.device, dtype)

    # noise handling
    x,h = model.augment_noise(x,one_hot,node_mask,charges)
    if len(model.conditioning) > 0:
        context = prepare_context(model.conditioning, 
                                  databatch, 
                                  model.property_norms).to(x.device, dtype)
        assert_correctly_masked(context, node_mask)
    else:
        context = None
    nll, reg_term, mean_abs_z = model.loss(x, h, node_mask,edge_mask, context)
    print(nll)

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

    