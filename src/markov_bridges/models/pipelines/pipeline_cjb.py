from markov_bridges.utils.paralellism import nametuple_to_device
from markov_bridges.utils.paralellism import check_model_devices
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

from markov_bridges.models.deprecated.generative_models.cjb_rate import ClassificationForwardRate
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.models.pipelines.samplers.tau_leaping_cjb import TauLeaping,TauLeapingOutput
from markov_bridges.models.pipelines.samplers.ode_solver_cfm import ODESamplerCFM
from markov_bridges.models.pipelines.abstract_pipeline import AbstractPipeline

class CJBPipeline(AbstractPipeline):
    """
    This is a wrapper for the stochastic process sampler TauDiffusion, will generate samples
    in the same device as the one provided for the forward model
    """
    def __init__(
            self,config:CMBConfig,
            rate_model:ClassificationForwardRate,
            dataloader:MarkovBridgeDataloader
            ):
        self.config:CMBConfig = config
        self.rate_model:ClassificationForwardRate = rate_model
        self.dataloder:MarkovBridgeDataloader = dataloader
        self.device = check_model_devices(self.rate_model)
        self.sampler = TauLeaping(config)
        
    def generate_sample(self,
                        databatch:MarkovBridgeDataNameTuple=None,
                        return_path:bool=False,
                        return_origin:bool=False)->TauLeapingOutput:
        """
        From an initial sample of databatch samples the data

        return_origin: deprecated 
        """
        databatch = nametuple_to_device(databatch,self.device)
        state = self.sampler.sample(self.rate_model,databatch,return_path=return_path)
        return state
            
    def __call__(self,
                 sample_size:int=100,
                 train:bool=True,
                 return_path:bool=False):
        """
        :param sample_size:
        :param train: If True sample initial points from train dataloader
        :param return_path: Return full path batch_size,number_of_time_steps,

        :return: MixedTauState
        """
        x_0 = self.dataloder.get_data_sample(sample_size,train=train)
        x_0 = nametuple_to_device(x_0, self.device)
        return self.generate_sample(x_0,return_path=return_path)