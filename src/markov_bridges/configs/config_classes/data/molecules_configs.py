from dataclasses import dataclass
from typing import Literal

@dataclass
class QM9Config:
    name:str="QM9"
    batch_size:int = 32
    num_workers:int = 12
    filter_n_atoms:int = None
    include_charges:bool = True
    subtract_thermo:bool = False
    force_download:bool = False
    remove_h:bool = False
    
    num_pts_train: int =  -1
    num_pts_test:int =  -1
    num_pts_valid:int  =  -1
    
    dataset:str = 'qm9'
    datadir:str = r"/home/piazza"
    wandb:bool = False
    
    context_node_nf:int=None
    vocab_size:int = None
    property_norms:dict = None
    
    has_target_discrete:int = True
    has_target_continuous:int = True
    continuos_dimensions:int = 3
    discrete_dimensions:int = None 
    
    context_node_nf:int=None
    vocab_size:int = None
    property_norms:dict = None
    
    has_target_discrete:int = True
    has_target_continuous:int = True
    continuos_dimensions:int = 3
    discrete_dimensions:int = None

#### make configuration class for LP dataset
@dataclass
class LPConfig:
    batch_size : int = 32 #batch size
    
    num_workers : int = 6 #number of subprocesses used to load the data (0 = sequantial data loading: may create a bottleneck)

    max_num_protein_nodes : int = 500 #max number of nodes a protein can have to pass the filter in the collate function of the dataloader
    accept_variable_bs : bool = False #STRONGLY RECOMMENDED:False. if true, the filter that discard instances where protein has more than max_num_protein_nodes nodes is done in the collate funciton (which can result in batches with different number of samples and possibly empty batches). If false, the filtering is done before so that the dataloader receives allready filtered instances

    padding_dependence : Literal["batch", "dataset"] = "batch" #RECOMMENDED: "batch". if "batch", the padding and the corresponding masks are created by the collate function of the dataloader, which means that everything is padded to the max number in the batch (in this way, different batches can have different shapes). If "dataset", the padding is performed with respect to the max number in the entire dataset (train, or valid or test), which means that each batch in each dataset has the same shape, buth the batch in the train has different shape from the batch in the test or valiation. 

    # !!! Is recommended to NOT put padding_dependence="dataset" & accept_variable_bs=True : too memory demanding

    num_pts_train : int = -1 #number of filtered instances to take from the entire training set. -1 means take all training filtered instances
    num_pts_valid : int = -1 #number of filtered instances to take from the entire validation set. -1 means take all filtered validation instances
    num_pts_test : int = -1 #number of filtered instances to take from the entire test set. -1 means take all filtered test instances

    shuffle_train : bool = True #shuffle to pass to the dataloader for train
    shuffle_valid : bool = False #shuffle to pass to the dataloader for valid
    shuffle_test : bool = False #shuffle to pass to the dataloader for test

    KEYS_TO_PAD = ["position_linker_gen", "category_linker_gen", #list of keys whose values need padding in the dataloader collate function
                   "position_fragment", "category_fragment", 
                   "position_protein", "category_protein",
                   "linker_edge_list", 
                   "fragment_edge_list", 
                   "protein_chopped_edge_list"] 
    

    train_path : str = "/home/piazza/markov_bridges/LP_Data/RestyledReducedDiffusionDataset/train.pt" #path for train set to load, preprocess and return via the dataloader
    valid_path : str = "/home/piazza/markov_bridges/LP_Data/RestyledReducedDiffusionDataset/validation.pt" #path for validation set to load, preprocess and return via the dataloader
    test_path : str = "/home/piazza/markov_bridges/LP_Data/RestyledReducedDiffusionDataset/test.pt" #path for test set to load, preprocess and return via the dataloader
     
