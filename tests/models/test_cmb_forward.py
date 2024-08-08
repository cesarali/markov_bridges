import torch
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network

from markov_bridges.data.categorical_samples import IndependentMixDataloader
from markov_bridges.models.generative_models.cmb_forward import MixedForwardMap
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig


if __name__=="__main__":

    model_config = CMBConfig()
    model_config.data = IndependentMixConfig(has_context_discrete=True)
    dataloader = IndependentMixDataloader(model_config.data)
    databatch = dataloader.get_databatch()
    cfm = MixedForwardMap(model_config,device=torch.device("cpu"))
    discrete_sample,continuous_sample = cfm.sample_bridge(databatch)
    rate,drift = cfm.forward_map(discrete_sample,continuous_sample,databatch.time,databatch)

    #loss = cfm.loss(databatch,discrete_sample,continuous_sample)
    #print(rate.shape)
    #print(loss)