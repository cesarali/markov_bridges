import torch
#import sys
#import os
# Add the directory containing the module to sys.path
#module_path = '/home/piazza/DiffusionModel/GitHub_GNN_LP_AM/ContextEncoder'
#if module_path not in sys.path:
#    sys.path.append(module_path)
#from LP_EGNN import Encoder
#import LP_EGNN as lpeg
#from markov_bridges.configs.config_classes.data.molecules_configs import LPConfig #dataset configuration
#from markov_bridges.data.dataloaders_utils import get_dataloaders 
#from markov_bridges.configs.experiments_configs.mixed.edmg_experiments import get_edmg_lp_experiment

"""
### NOTE just as a temp the instance of the model is defined here , but in the real implementatio it will be defined probably somewhere else and passed to the function
config = get_edmg_lp_experiment() #call the config file
config.data = LPConfig(batch_size=128, train_path="/home/piazza/EncoderLinkerSizePredData/train.pt", test_path="/home/piazza/EncoderLinkerSizePredData/test.pt", valid_path="/home/piazza/EncoderLinkerSizePredData/validation.pt")
dataloaders = get_dataloaders(config) #dataloaders
dataset_info = dataloaders.dataset_info #dataset info
model = lpeg.Encoder(dataset_info) #!!! model instance NOTE remember to change IsLinkerSizePred and isLinkerSizePred_Classifier both to false in the config
"""

def prepare_context(batch, model):
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

