from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network

from markov_bridges.data.categorical_samples import IndependentMixDataloader
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from markov_bridges.models.generative_models.cmb_forward import MixedForwardMap
from markov_bridges.models.pipelines.samplers.mixed_tau_diffusion import TauDiffusion
from markov_bridges.models.pipelines.pipeline_cmb import CMBPipeline

import torch


if __name__=="__main__":
    model_config = CMBConfig()
    model_config.data = IndependentMixConfig(has_context_continuous=True)
    dataloader = IndependentMixDataloader(model_config.data)
    databatch = dataloader.get_databatch()
    cfm = MixedForwardMap(model_config,device=torch.device("cpu"),join_context=dataloader.join_context)
    pipeline = CMBPipeline(model_config,cfm,dataloader)
    
    for databatch in dataloader.test():
        generative_sample = pipeline.generate_sample(databatch)
        
    #sampler = TauDiffusion(model_config,dataloader.join_context)
    #state, ts = sampler.sample(cfm,databatch,return_path=False)
    #print(state.discrete.shape)