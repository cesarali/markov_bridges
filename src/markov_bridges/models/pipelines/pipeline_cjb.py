import torch
from markov_bridges.utils.paralellism import check_model_devices
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

from markov_bridges.models.pipelines.samplers.tau_leaping_cjb import TauLeapingCJB
from markov_bridges.data.utils import sample_from_dataloader_iterator
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.models.generative_models.cjb_rate import ClassificationForwardRate

from dataclasses import dataclass

@dataclass
class CJBPipelineOutput:
    raw_sample:torch.tensor = None
    path_histogram:torch.tensor = None
    path_time:torch.tensor = None
    x_0:MarkovBridgeDataNameTuple = None
     
class CJBPipeline:
    """
    """
    def __init__(self,config:CJBConfig,rate_model:ClassificationForwardRate,dataloader):
        self.config:CJBConfig = config
        self.rate_model = rate_model
        self.dataloder:MarkovBridgeDataloader = dataloader
        self.device = check_model_devices(self.rate_model)

    def generate_sample(self,
                        x_0:MarkovBridgeDataNameTuple=None,
                        return_path:bool=False,
                        return_intermediaries:bool=False,
                        return_origin:bool=False)->CJBPipelineOutput:
        """
        From an initial sample of x_0 samples the data
        """
        x_f, x_hist, x0_hist, ts = TauLeapingCJB(self.config,self.rate_model,x_0,return_path=return_path)
        # Return results based on flags
        if return_path or return_intermediaries:
            if return_origin:
                return CJBPipelineOutput(raw_sample=x_f,x_0=x_0,path_histogram=x_hist,path_time=ts)
            else:
                return CJBPipelineOutput(raw_sample=x_f,path_histogram=x_hist,path_time=ts)
        else:
            if return_origin:
                return CJBPipelineOutput(raw_sample=x_f,x_0=x_0)
            else:
                return CJBPipelineOutput(raw_sample=x_f)
            
    def __call__(self,
                 sample_size:int=100,
                 train:bool=True,
                 return_path:bool=False,
                 return_intermediaries:bool=False,
                 return_origin:bool=False)->CJBPipelineOutput:
        """
        :param sample_size:
        :param train: If True sample initial points from  train dataloader
        :param return_path: Return full path batch_size,number_of_time_steps,
        :param return_intermediaries: Return path only at intermediate points

        :return: x_f, last sampled point
                 x_path, full diffusion path
                 t_path, time steps in temporal path
        """
        x_0 = self.dataloder.get_data_sample(sample_size,train=train)
        return self.generate_sample(x_0,
                                    return_path=return_path,
                                    return_intermediaries=return_intermediaries,
                                    return_origin=return_origin)