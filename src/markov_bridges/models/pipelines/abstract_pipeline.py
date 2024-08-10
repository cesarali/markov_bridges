from abc import ABC, abstractmethod

class AbstractPipeline(ABC):
    """
    This is a wrapper for the stochastic process sampler TauDiffusion, which will generate samples
    on the same device as the one provided for the forward model.
    """
    
    def __init__(self, config, generation_model, dataloader):
        self.config = config
        self.generation_model = generation_model
        self.dataloder = dataloader
        self.sampler = None  # This should be set in the concrete implementation
    
    @abstractmethod
    def generate_sample(self, databatch=None, return_path=False):
        """
        Generates a sample from the stochastic process.

        :param databatch: Initial batch of data for sampling.
        :param return_path: If True, return the full path.
        :param return_origin: (Deprecated) Return the origin.
        :return: MixedTauState
        """
        pass
    
    @abstractmethod
    def __call__(self, sample_size=100, train=True, return_path=False):
        """
        Invokes the pipeline to generate samples.

        :param sample_size: Number of samples to generate.
        :param train: If True, sample initial points from the train dataloader.
        :param return_path: If True, return the full path.
        :return: MixedTauState
        """
        pass