from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from markov_bridges.configs.config_classes.pipelines.pipeline_configs import CMBPipelineConfig

from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network

from markov_bridges.data.categorical_samples import IndependentMixDataloader
from markov_bridges.models.deprecated.generative_models.cmb_forward import MixedForwardMap
from markov_bridges.models.pipelines.pipeline_cmb import CMBPipeline

import torch


def test_mixed_tau():
    model_config = CMBConfig(continuous_loss_type="flow")
    model_config.data = IndependentMixConfig()
    model_config.pipeline = CMBPipelineConfig(solver="ode_tau")

    dataloader = IndependentMixDataloader(model_config.data)
    databatch = dataloader.get_databatch()
    cfm = MixedForwardMap(model_config,device=torch.device("cpu"))
    pipeline = CMBPipeline(model_config,cfm,dataloader)
    
    for databatch in dataloader.test():
        generative_sample = pipeline.generate_sample(databatch,return_path=True)
        break    
    print(generative_sample.discrete_paths.shape)

if __name__=="__main__":

    test_mixed_tau()

    #sampler = TauDiffusion(model_config,dataloader.join_context)
    #state, ts = sampler.sample(cfm,databatch,return_path=False)
    #print(state.discrete.shape)