import torch
from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMGConfig
from markov_bridges.models.networks.temporal.edmg.egnn_dynamics import EGNN_dynamics_QM9
from markov_bridges.models.networks.temporal.edmg.en_diffusion import EnVariationalDiffusion
from markov_bridges.models.networks.temporal.edmg.helper_distributions import DistributionNodes,DistributionProperty

def get_edmg_model(config:EDMGConfig, dataset_info, dataloader_train, device=None):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(config.data.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(config.noising_model.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, config.noising_model.conditioning)

    if config.noising_model.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, 
        context_node_nf=config.noising_model.context_node_nf,
        n_dims=3, 
        device=device, 
        hidden_nf=config.noising_model.nf,
        act_fn=torch.nn.SiLU(), 
        n_layers=config.noising_model.n_layers,
        attention=config.noising_model.attention, 
        tanh=config.noising_model.tanh, 
        mode=config.noising_model.model, 
        norm_constant=config.noising_model.norm_constant,
        inv_sublayers=config.noising_model.inv_sublayers, 
        sin_embedding=config.noising_model.sin_embedding,
        normalization_factor=config.noising_model.normalization_factor, 
        aggregation_method=config.noising_model.aggregation_method)

    if config.noising_model.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=config.noising_model.diffusion_steps,
            noise_schedule=config.noising_model.diffusion_noise_schedule,
            noise_precision=config.noising_model.diffusion_noise_precision,
            loss_type=config.noising_model.diffusion_loss_type,
            norm_values=config.noising_model.normalize_factors,
            include_charges=config.data.include_charges
            )
        return vdm,nodes_dist, prop_dist
    else:
        raise ValueError(config.noising_model.probabilistic_model)
    