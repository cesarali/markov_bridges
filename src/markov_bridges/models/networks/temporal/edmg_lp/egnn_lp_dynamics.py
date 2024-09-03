import torch
import numpy as np
import torch.nn as nn
from markov_bridges.models.networks.temporal.edmg_lp.graph_gnn_lp import GNN,EGNN
from markov_bridges.utils.equivariant_diffusion import remove_mean, remove_mean_with_mask
from dataclasses import fields, asdict
from dataclasses import  is_dataclass
from dataclasses import dataclass
from typing import Dict, Any, ClassVar

import sys
# Add the directory containing the files to the Python path
sys.path.append('/home/piazza/DiffusionModel/GitHub_GNN_LP_AM/ContextEncoder')
from LP_EGNN import Encoder
from EncoderConfigs import EncoderConfig
#sys.exit()

def get_encoder_config(device, hidden_layers_shape=None, output_shape=None):
    """
    Function to get the correct configuration for the context encoder.

    Parameters:
    -----------
    device : str
        device for all 3 blocks of the encoder (for example: 'cuda:5')
    hidden_layers_shape : list (optional, default=None)
        list with hidden layers shape
    output_shape : int (optional, default=None)
        number of output neurons; this is also the lenght of the context embedding for each instance (aka. context_node_nf)

    Returns:
    --------
    context_encoder_config : config class for context encoder with updated settings
    """
    context_encoder_config = EncoderConfig #reference to the class for encoder config with custom settings for context encoder
    context_encoder_config.mlp_encoder_config.isLinkerSizePred = False #not used for regression on linker size pred
    context_encoder_config.mlp_encoder_config.isLinkerSizePred_Classifier = False #not used for classification on linker size pred
    context_encoder_config.mlp_encoder_config.device = device #set device to the passed device in the function for mlp final encoder
    context_encoder_config.protein_gnn_config.device  = device #set device to the passed device in the function for protein gnn block
    context_encoder_config.fragment_gnn_config. device = device #set device to the passed device in the function for gragment gnn bock
    if hidden_layers_shape is not None: #if you specify a custom hydden layers shape
        context_encoder_config.mlp_encoder_config.hidden_layers_shape = hidden_layers_shape
    if output_shape is not None: #if you specfy a custom output shape (in this case this is the shape of the embedding!!)
        context_encoder_config.mlp_encoder_config.out_shape = output_shape
    return context_encoder_config

"""
DO NOT DELETE: THIS IS A SNIPPET TO CHECK IF THE get_encoder_config FUNCTION WORKS

test = get_encoder_config(device="cuda:0", hidden_layers_shape=[10,20,10], output_shape=5) #create a custom config

def test_get_encoder_config(conf):
    ## This function is just to test that the get_encoder_config function works and actually returns a config with the specified settings
    for class_name, classe in conf.__annotations__.items():
        print("-"*40)
        print(class_name) #this is one of the three fields name in the EncoderConfig; classe is instead the class associated to the field
        print("-"*40)
        for field_name, field_value in classe.__annotations__.items(): #each of the three class has itself other fields associatd to certain values
            attribute = getattr(classe, field_name) #get from the class the value associated t the field with name field_name
            print(field_name," : " , attribute)

test_get_encoder_config(test) #print the new custom config returned by the get_encoder_config function

sys.exit()
"""


class EGNN_dynamics_LP(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, ##NOTE: whatever value you pass in the init of the class for context_node_nf this will then be overridden with the correct value for LP dataset according to the context encoder
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super().__init__()

        ## define context encoder configuration and the correct number of context features according to the encoder configuration
        self.context_encoder_config = get_encoder_config(device, hidden_layers_shape=[256,128,64], output_shape=32) #TODO put a custom number of hidden layers and output shape, those here are just examples
        context_node_nf = self.context_encoder_config.mlp_encoder_config.out_shape #directly from the config retrieve the context_node_nf (so whatever you pass in the init of the dynamics, internally this is overridden and set to the correct value for LP data)
        self.context_node_nf = context_node_nf #set correct value also to self.context_node_nf

        #print("normal ", context_node_nf)
        #print("self ", self.context_node_nf)

        self.mode = mode
        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics': #unused in my case, but not removed just in case
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time 
         

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward
    
    def prepare_context(self,batch, model):
        """
        Function to prepare the context for conditioning the generative model.

        Parameters:
        -----------
        batch : 
            a batch of instances given by the LPDataloader 
        model :
            the model that creates the embedded representation of the context

        Returns:
        --------
        expanded_context_masked : torch.Tensor of shape [batch size, num linker nodes, num context features]
            a tensor where for each linker node in each instance batch we have the embedded context. 
            Of course, all the linker nodes of the same instance will share the same context.
            Additionally, since not all instances in the batch really have the same number of nodes and some nodes are just pad, the context returned will be masked 
            in such a way that it will be a tensor of all zeros for padded nodes. 
        """
        # assuming that the model is defined, the model outputs is [batch size, number of context features]: this is a batch of embedded contextes
        embedded_contextes_batch = model(batch) #pass the batch to the model and obtain the context embedding for each instance in the batch. embedded_contextes_batch has shape [batch size, output shape]
        linker_atom_mask = batch["mask_category_linker_gen"] #this is a mask that define, for each instance in the batch, if a linker node exists or it is a pad. it has shape [bs, num linker nodes]. We could have taken also mask_position_linker_gen, it would have been the same
        bs, num_linker_nodes = linker_atom_mask.shape[0], linker_atom_mask.shape[1] #from the mask tensor retrieve batch size and number of linker nodes inthe batch

        embedded_contextes_batch = embedded_contextes_batch.unsqueeze(1) #Unsqueeze to add a new dimension for linker nodes: Shape is now: [batch_size, 1, context_features]
        expanded_context = embedded_contextes_batch.repeat(1, num_linker_nodes, 1) #Repeat the context across the num_linker_nodes dimension. Shape is now: [batch_size, num_linker_nodes, context_features]

        # mask the expanded context: not all the linker nodes are real, some of them are pad and we know them thanks to the linker_atom_mask. Using the same mask, mask also the context tensor in such a way that the context associated to pad nodes is a tensor full of 0
        linker_atom_mask = linker_atom_mask.unsqueeze(-1) #Unsqueeze the mask to make it compatible for broadcasting: not it has shape [batch_size, num_linker_nodes, 1]
        expanded_context_masked = expanded_context * linker_atom_mask #multiply the context tensor for the mask tensor (context shape: [batch_size, num_linker_nodes, context_features], mask shape: [batch_size, num_linker_nodes, 1])

        return expanded_context_masked #[bs, num linker nodes, num context features] masked, i.e., the context for a node that does not exist is just a tensor full of 0


    def _forward(self, t, xh, node_mask, edge_mask, batch, dataset_info):
        """
        t : timestep
        xh : node coordinates and features concatenated for a batch of linker atoms
        node mask : node mask
        edge mask : edge mask
        batch : batch of instances (when calling the _forward it will be set to the batch returned by the dataloader):
            required because the encoder will take this batch to crate the context embedding
        dataset_info : dataset info returned by dataloaders and required by the encoder
        """
        bs, n_nodes, dims = xh.shape #n_nodes is in our case the number of linker ndes
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, t.device)
        edges = [x.to(t.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(t.device)
        else:
            h = xh[:, self.n_dims:].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        ## define the context encoder, which is the model that takes a batch of instances provided by the dataloader and returns the embedded context for each instance in the batch
        context_encoder = Encoder(dataset_info, self.context_encoder_config) #define the model instance by passing the dataset info and the configuration settings

        ## using the model and the batch of instances retrieve the context as required by the code flow (context is [bs, num linker node, num context features] and is masked with linker node mask)
        context = self.prepare_context(batch, context_encoder) #[bs, num linker node, num context features] , already masked

        #if context:
        # We're conditioning, awesome!
        context = context.view(bs*n_nodes, self.context_node_nf)
        h = torch.cat([h, context], dim=1)

        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        #if context:
        # Slice off context size:
        h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

#istance = EGNN_dynamics_LP(1, 12, 3) #temp to see if context node nf is set to correct value