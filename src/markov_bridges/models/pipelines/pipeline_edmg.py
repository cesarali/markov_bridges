import numpy as np

import torch
import torch.nn.functional as F

from markov_bridges.data.qm9.sampling import sample_chain

from markov_bridges.utils.paralellism import nametuple_to_device
from markov_bridges.utils.paralellism import check_model_devices
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from markov_bridges.models.pipelines.abstract_pipeline import AbstractPipeline

from markov_bridges.data.qm9.qm9_points_dataloader import QM9PointDataloader
from markov_bridges.models.networks.temporal.edmg.en_diffusion import EnVariationalDiffusion
from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMGConfig

from markov_bridges.utils.equivariant_diffusion import (
    assert_mean_zero_with_mask, 
    remove_mean_with_mask,
    assert_correctly_masked,
    reverse_tensor
)
from markov_bridges.data.qm9.analyze import check_stability
        
class EDGMPipeline(AbstractPipeline):
    """
    This is a wrapper for the stochastic process sampler TauDiffusion, will generate samples
    in the same device as the one provided for the forward model
    """
    denoising_model:EnVariationalDiffusion

    def __init__(
            self,
            config:EDMGConfig,
            model,
            dataloader:QM9PointDataloader,
            ):
        
        self.config:EDMGConfig = config
        self.noising_model = model.noising_model
        self.dataloder:MarkovBridgeDataloader = dataloader
        self.dataset_info = self.dataloder.dataset_info
        self.prop_dist = model.prop_dist
        self.nodes_dist = model.nodes_dist
        
    def generate_sample(self,
                        databatch:MarkovBridgeDataNameTuple=None,
                        return_path:bool=False,
                        return_origin:bool=False):
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

    def sample_chain(
            self,
            n_tries, 
        ):

        flow = self.noising_model
        device = check_model_devices(flow)
        dataset_info = self.dataset_info
        prop_dist = self.prop_dist
    
        n_samples = 1
        if self.config.data.dataset == 'qm9' or self.config.data.dataset == 'qm9_second_half' or self.config.data.dataset == 'qm9_first_half':
            n_nodes = 19
        elif self.config.data.dataset == 'geom':
            n_nodes = 44
        else:
            raise ValueError()

        # TODO FIX: This conditioning just zeros.
        if self.config.noising_model.context_node_nf > 0:
            context = prop_dist.sample(n_nodes).unsqueeze(1).unsqueeze(0)
            context = context.repeat(1, n_nodes, 1).to(device)
            #context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
        else:
            context = None

        node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
        edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = flow.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100)
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

            # Prepare entire chain.
            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(torch.argmax(one_hot, dim=2), num_classes=len(dataset_info['atom_decoder']))
            charges = torch.round(chain[:, :, -1:]).long()

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')
        return one_hot, charges, x

    def sample(
            self,
            nodesxsample=torch.tensor([10]), 
            context=None,
            fix_noise=False
        ):
        generative_model = self.noising_model
        device = check_model_devices(generative_model)
        dataset_info = self.dataset_info
        prop_dist = self.prop_dist

        max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

        assert int(torch.max(nodesxsample)) <= max_n_nodes
        batch_size = len(nodesxsample)

        node_mask = torch.zeros(batch_size, max_n_nodes)
        for i in range(batch_size):
            node_mask[i, 0:nodesxsample[i]] = 1

        # Compute edge_mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
        node_mask = node_mask.unsqueeze(2).to(device)

        # TODO FIX: This conditioning just zeros.
        if self.config.noising_model.context_node_nf > 0:
            if context is None:
                context = prop_dist.sample_batch(nodesxsample)
            context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
        else:
            context = None

        x, h = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise)

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)
        one_hot = h['categorical']
        charges = h['integer']

        assert_correctly_masked(one_hot.float(), node_mask)
        if self.config.data.include_charges:
            assert_correctly_masked(charges.float(), node_mask)
        return one_hot, charges, x, node_mask

    def sample_sweep_conditional(
            self,
            n_nodes=19, 
            n_frames=100
        ):
        generative_model = self.noising_model
        device = check_model_devices(generative_model)
        dataset_info = self.dataset_info
        prop_dist = self.prop_dist

        nodesxsample = torch.tensor([n_nodes] * n_frames)
        context = []
        for key in prop_dist.distributions:
            min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
            mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
            min_val = (min_val - mean) / (mad)
            max_val = (max_val - mean) / (mad)
            context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
            context.append(context_row)
        context = torch.cat(context, dim=1).float().to(device)

        one_hot, charges, x, node_mask = self.sample(nodesxsample=nodesxsample, context=context, fix_noise=True)
        return one_hot, charges, x, node_mask
    
def save_and_sample_chain(pipeline:EDGMPipeline):
    one_hot, charges, x = pipeline.sample_chain(n_tries=1)
    return one_hot, charges, x

def save_and_sample_conditional(pipeline:EDGMPipeline, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = pipeline.sample_sweep_conditional()
    #vis.save_xyz_file(
    #    'outputs/%s/epoch_%d/conditional/' % (config.exp_name, epoch), one_hot, charges, x, dataset_info,
    #    id_from, name='conditional', node_mask=node_mask)
    return one_hot, charges, x

def sample_different_sizes_and_save(pipeline:EDGMPipeline,n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = pipeline.nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = pipeline.sample(nodesxsample=nodesxsample)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        #vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
        #                  batch_size * counter, name='molecule')