import torch
from markov_bridges.utils.paralellism import check_model_devices
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from markov_bridges.models.pipelines.samplers.ode_solver_cfm import ODESamplerCFM
from markov_bridges.configs.config_classes.generative_models.cfm_config import CFMConfig
from markov_bridges.models.generative_models.cfm_forward import ContinuousForwardMap

from markov_bridges.utils.paralellism import nametuple_to_device

from dataclasses import dataclass

@dataclass
class CFMPipelineOutput:
    trajectories: torch.tensor = None
    context_discrete: torch.tensor = None
    context_continuous: torch.tensor = None
     
class CFMPipeline:
    """
    """
    def __init__(self, config: CFMConfig, drift_model: ContinuousForwardMap, dataloader):
        self.config: CFMConfig = config
        self.drift_model = drift_model
        self.dataloder: MarkovBridgeDataloader = dataloader
        self.device = check_model_devices(self.drift_model)

    def generate_sample(self,
                        x_0: MarkovBridgeDataNameTuple=None,
                        return_path: bool=False) -> CFMPipelineOutput:
        """
        From an initial sample of x_0 samples the data
        """
        x_0 = nametuple_to_device(x_0, self.device)
        trajectories = ODESamplerCFM(self.config, self.drift_model, x_0, return_path=return_path)
        trajectories = trajectories.permute(1,0,2)
        context_discrete = x_0.context_discrete if self.config.data.has_context_discrete else None
        context_continuous = x_0.context_continuous if self.config.data.has_context_continuous else None

        return CFMPipelineOutput(trajectories=trajectories,  
                                 context_discrete=context_discrete, 
                                 context_continuous=context_continuous )

            
    def __call__(self,
                 sample_size:int=100,
                 return_path:bool=False)->CFMPipelineOutput:
        """
        :param sample_size:
        :param train: If True sample initial points from  train dataloader
        :param return_path: Return full path batch_size,number_of_time_steps,
        :param return_intermediaries: Return path only at intermediate points

        :return: x_f, last sampled point
                 x_path, full diffusion path
                 t_path, time steps in temporal path
        """
        x_0 = self.dataloder.get_data_sample(sample_size, train=False)
        x_0 = nametuple_to_device(x_0, self.device)
        return self.generate_sample(x_0, return_path=return_path)

    