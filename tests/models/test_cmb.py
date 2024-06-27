from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig

from markov_bridges.models.generative_models.cmb import CMB
import torch

if __name__=="__main__":
    model_config = CMBConfig()
    model_config.data = IndependentMixConfig(has_context_discrete=True)

    cmb = CMB(model_config,device=torch.device("cpu"))
    databatch = cmb.dataloader.get_databatch()
    discrete_sample,continuous_sample = cmb.forward_map.sample_bridge(databatch)
    loss = cmb.forward_map.loss(databatch,discrete_sample,continuous_sample)
    print(loss)
