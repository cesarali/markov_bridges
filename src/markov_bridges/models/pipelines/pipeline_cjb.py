import torch
from markov_bridges.utils.paralellism import check_model_devices
from markov_bridges.models.pipelines.samplers.tau_leaping import TauLeaping
from markov_bridges.data.utils import sample_from_dataloader_iterator
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig

class CJBPipeline:
    """
    """
    def __init__(self,config:CJBConfig,model,dataloader_0,dataloader_1,parent_dataloader=None):
        self.model = model
        self.config = config
        self.dataloder_0 = dataloader_0
        self.dataloder_1 = dataloader_1
        self.parent_dataloader = parent_dataloader


        self.model = model
        self.device = check_model_devices(self.model)

    def get_x0_sample(self,sample_size,train=True,origin=False):
        if not origin:
            # select the right iterator
            if hasattr(self.dataloder_0, "train"):
                dataloder_iterator = self.dataloder_0.train() if train else self.dataloder_0.test()
            else:
                dataloder_iterator = self.dataloder_0
            x_0 = sample_from_dataloader_iterator(dataloder_iterator, sample_size)

            return x_0,None
        else:
            if hasattr(self.dataloder_0, "train"):
                dataloder_iterator = self.dataloder_0.train() if train else self.dataloder_0.test()
            else:
                dataloder_iterator = self.dataloder_0
            x_0 = sample_from_dataloader_iterator(dataloder_iterator, sample_size)

            if hasattr(self.dataloder_1, "train"):
                dataloder_iterator = self.dataloder_1.train() if train else self.dataloder_1.test()
            else:
                dataloder_iterator = self.dataloder_1

            x_1 = sample_from_dataloader_iterator(dataloder_iterator, sample_size)
            return x_0,x_1


    def __call__(self,
                 sample_size:bool=100,
                 train:bool=True,
                 return_path:bool=False,
                 return_intermediaries:bool=False,
                 batch_size:int=128,
                 x_0:torch.Tensor=None,
                 origin:bool=False,
                 repeat_sample:int=0):
        """
        For Conditional Rate Matching We Move Forward in Time

        :param sample_size:
        :param train:   If True sample initial points from  train dataloader
        :param return_path:   Return full path batch_size,number_of_time_steps,
        :param return_intermediaries:  Return path only at intermediate points
        :param batch_size: Maximum size of the batch to create process
        :param x_0:  If given, uses this point as start of generative procedure

        :return: x_f, last sampled point
                 x_path, full diffusion path
                 t_path, time steps in temporal path
        """
        if return_intermediaries:
            return_path = False

        # Get the initial sample
        if x_0 is None:
            x_0,x_original = self.get_x0_sample(sample_size=sample_size, train=train,origin=origin)
            x_0 = x_0.to(self.device)
            if x_original is not None:
                x_original = x_original.to(self.device)
        else:
            batch_size = x_0.size(0)
            x_0 = x_0.view(batch_size,-1)
        
        if repeat_sample > 0:
            x_0 = x_0.repeat_interleave(repeat_sample,dim=0)
            if origin:
                x_original = x_original.repeat_interleave(repeat_sample,dim=0)

        # If batch_size is not set or sample_size is within the batch limit, process normally
        if batch_size is None or sample_size <= batch_size:
            if hasattr(self.config.data1, "conditional_model"):
                if self.config.data1.conditional_model:
                    x_f, x_hist, x0_hist, ts = TauLeaping(self.config, self.model, x_0, forward=True,return_path=return_path)
            else:
                x_f, x_hist, x0_hist, ts = TauLeaping(self.config, self.model, x_0, forward=True, return_path=return_path)
        else:
            # Initialize lists to store results from each batch
            x_f_batches = []
            x_hist_batches = []
            x0_hist_batches = []

            remaining = x_0.size(0)
            while remaining > 0:
                # Determine the number of elements to process
                batch_size = min(remaining, batch_size)

                # Slice the data into a batch
                x_0_batch = x_0[:batch_size]
                x_f_batch, x_hist_batch, x0_hist_batch, ts = TauLeaping(self.config, 
                                                                        self.model, 
                                                                        x_0_batch,
                                                                        forward=True, 
                                                                        return_path=return_path)

                # Append the results from the batch
                x_f_batches.append(x_f_batch)
                if return_path or return_intermediaries:
                    x_hist_batches.append(x_hist_batch)
                    x0_hist_batches.append(x0_hist_batch)
                x_0 = x_0[batch_size:]
                remaining -= batch_size

            # Concatenate the results from all batches
            x_f = torch.cat(x_f_batches, dim=0)
            if return_path or return_intermediaries:
                x_hist = torch.cat(x_hist_batches, dim=0)
                x0_hist = torch.cat(x0_hist_batches, dim=0)

        # Return results based on flags
        if return_path or return_intermediaries:
            if origin:
                return x_f, x_original,x_hist, ts
            else:
                return x_f,x_hist,ts
        else:
            if origin:
                return x_f,x_original
            else:
                return x_f

