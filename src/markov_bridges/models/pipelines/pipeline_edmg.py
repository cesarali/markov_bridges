from markov_bridges.data.qm9.sampling import sample_chain

from markov_bridges.utils.paralellism import nametuple_to_device
from markov_bridges.utils.paralellism import check_model_devices
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMGConfig
from markov_bridges.models.pipelines.samplers.tau_leaping_cjb import TauLeaping,TauLeapingOutput
from markov_bridges.models.pipelines.abstract_pipeline import AbstractPipeline

def save_and_sample_chain(model, args, device, dataset_info, prop_dist):
    one_hot, charges, x = sample_chain(config=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)
    return one_hot, charges, x

class EDGMPipeline(AbstractPipeline):
    """
    This is a wrapper for the stochastic process sampler TauDiffusion, will generate samples
    in the same device as the one provided for the forward model
    """
    def __init__(
            self,config:EDMGConfig,
            rate_model,
            dataloader:MarkovBridgeDataloader
            ):
        self.config:EDMGConfig = config
        self.rate_model = rate_model
        self.dataloder:MarkovBridgeDataloader = dataloader
        self.device = check_model_devices(self.rate_model)
        
    def generate_sample(self,
                        databatch:MarkovBridgeDataNameTuple=None,
                        return_path:bool=False,
                        return_origin:bool=False)->TauLeapingOutput:
        """
        From an initial sample of databatch samples the data

        return_origin: deprecated
        """
        pass
            
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
        pass