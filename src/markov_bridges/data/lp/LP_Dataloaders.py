# dataloader for lp dataset


##import torch
#from torch.utils.data import Dataset

#define custom class for the dataloader
#class LPDataset(Dataset):
#    def __init__(self, data):
#        self.data = data#
#
#    def __len__(self):
#        return len(self.data)
#
#    def __getitem__(self, idx):
#        return self.data[idx]
    

    
#def _collate(self, dataset):
#    """
#    Custom collate function. 
#    It takes as input a dataset (can be train, validation or test set), and rearrange informations.#
#
#    dataset is a list of dictionaries like [{"mol_number":1, "mol_pos":torch.tensor with positions of molecule 1, "mol_onehot":torch.tensor with onehot of molecule one}, {same for mol2..}]
#    and returns a dictionary like {"mol_number": torch.tensor of all molecules number, "mol_pos":torch.tensor of all molecule atoms positions} ...
#    """
#    out = {} #intialize out dictionary
#    for sample in dataset: #for each sample in the dataset
#        for key, value in sample.items(): #iterate through keys
#            out.setdefault(key, []).append(value) #set keys to out dictionary if not yet existing, then append values to the key
    
    
    #padding

    
#def get_dataloaders(self, train_path, val_path, test_path, batch_size):
#    train, val, test = torch.load(train_path), torch.load(val_path), torch.load(test_path) #load .pt files -> obtain list of dictionaries where every dictionary is for a sample











from dataclasses import dataclass
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from torch.utils.data import DataLoader
import torch

#### make configuration class for LP dataset
@dataclass
class LPConfig:
    batch_size : int = 32 
    num_workers : int = 12

    num_pts_train : int = -1 #by default take all training instances
    num_pts_valid : int = -1 #by default take all validation instances
    num_pts_test : int = -1 #by default take all test instances

    #shuffle_train : bool = True #shuffle to pass to the dataloader for train
    #shuffle_valid : bool = False #shuffle to pass to the dataloader for valid
    #shuffle_test : bool = False #shuffle to pass to the dataloader for test

    KEYS_TO_PAD = ["position_linker_gen", "category_linker_gen", #list of keys whose values need padding in the dataloader collate function
                   "position_fragment", "category_fragment", 
                   "position_protein", "category_protein",
                   "linker_edge_list", 
                   "fragment_edge_list", 
                   "protein_chopped_edge_list"] 
    

    train_path : str = "/home/piazza/markov_bridges/LP_Data/RestyledReducedDiffusionDataset/train.pt" #path for train set to load, preprocess and return via the dataloader
    valid_path : str = "/home/piazza/markov_bridges/LP_Data/RestyledReducedDiffusionDataset/validation.pt" #path for validation set to load, preprocess and return via the dataloader
    test_path : str = "/home/piazza/markov_bridges/LP_Data/RestyledReducedDiffusionDataset/test.pt" #path for test set to load, preprocess and return via the dataloader


class LPDataloader(MarkovBridgeDataloader):
    #dataset_config : LPConfig

    def __init__(self, which_dataset):
        self.which_dataset = which_dataset #which subset you want (train, valid or test)
        self.dataset_config = LPConfig 
        #self.dataloader = self.get_dataloader(self.dataset)
        

    def load_subset(self): #function to load all 3 .pt files
        """
        Load train, validation and test set original .pt files, which are list of dictionaries where each dictioary contains information for an instance.
        For each set obtain the entire list or a part of it according to the desred number of instances to retrieve from each set, 
        then return a dictionary with the obtained train, validation and test list of instances.

        Returns:
        --------
        {"train": selected_train,
                "valid": selected_valid,
                "test": selected_test}
        """
        if self.which_dataset == "train":
            train = torch.load(self.dataset_config.train_path)
            selected_dataset = train if self.dataset_config.num_pts_train == -1 else train[:self.dataset_config.num_pts_train] #select training instances to be returned in batch by the dataloader (all or a certain number specified in num_pts_train)
        elif self.which_dataset == "valid":
            valid = torch.load(self.dataset_config.valid_path)
            selected_dataset = valid if self.dataset_config.num_pts_valid == -1 else valid[:self.dataset_config.num_pts_valid] #select validation instances
        elif self.which_dataset == "test":
            test = torch.load(self.dataset_config.test_path)
            selected_dataset = test if self.dataset_config.num_pts_test == -1 else test[:self.dataset_config.num_pts_test] #select test instances

        #train, valid, test = torch.load(self.dataset_config.train_path), torch.load(self.dataset_config.valid_path), torch.load(self.dataset_config.test_path) #load original train validation and test
        #selected_train = train if self.dataset_config.num_pts_train == -1 else train[:self.dataset_config.num_pts_train] #select training instances to be returned in batch by the dataloader (all or a certain number specified in num_pts_train)
        #selected_valid = valid if self.dataset_config.num_pts_valid == -1 else valid[:self.dataset_config.num_pts_valid] #select validation instances
        #selected_test = test if self.dataset_config.num_pts_test == -1 else test[:self.dataset_config.num_pts_test] #select test instances
        #return {"train": selected_train,
        #        "valid": selected_valid,
        #        "test": selected_test}
        return selected_dataset


    #def processing(self):
    #    preprocessed_subsets = self.load_subsets() #dictionary with train, validation and test sets selected instances
    #    for subset in preprocessed_subsets:

    def custom_collate(self, data):
        """
        Custom collate function. 

        It takes as input a subset (can be train, validation or test set) returned by the load_subset function, rearrange informations, performs padding and create masks after padding.

        subset is a list of dictionaries like [{"mol_number":1, "mol_pos":torch.tensor with positions of molecule 1, "mol_onehot":torch.tensor with onehot of molecule one}, {same for mol2..}]:
        it contains all instances for a given subset (train, valid or test).

        custom_collate first creates the out dictionary to be like {"mol_number": torch.tensor of all molecules number, "mol_pos":torch.tensor of all molecule atoms positions} ...,
        then padd all values specified in config KEYS_TO_PAD, and finally create the masks for padded things.
        """

        #### restyle data in the form of dictionay with unique keys, where to each key there are associated values of all instances
        out = {} #intialize out dictionary: this dictionary will contain unique keys and all the subset values
        for sample in data: #for each sample in the subset
            for k, v in sample.items(): #iterate through keys and values
                out.setdefault(k, []).append(v) #set keys to out dictionary if not yet existing, then append values to the key

        #### padding
        for key, value in out.items(): #iterate through keys and values of out dictionary
            if key in self.dataset_config.KEYS_TO_PAD:  #if the value need to be padded
                if "edge" not in key: #if it is not theedge list
                    if value[0].dtype != torch.float64: #check the dtype of the first tensor, if not double
                        value = [tensor.to(torch.float64) for tensor in value] #convert all tensors in the list to torch.float64
                else: #if it is an edge list
                    value = [torch.stack(element).to(torch.float64).transpose(0,1) for element in value] #take each edge list stored in value, stack the two torch tensors (sender and receiver), convert to torch double dtype and reshape to be [N edges, 2]
                out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=float("inf")) #pad with inf (recognizable padding for later mask)

        #### create masks after padding. for each of the three categories (linker, fragment, protein) we need a mask for positions, one for category and one for edges
        ## positions masks
        mask_position_linker_gen = torch.zeros(size=(out["position_linker_gen"].shape[0], out["position_linker_gen"].shape[1])) #tensor full of 0 with shape [number of samples in dataset, max num of linker atoms]
        for i in range(out["position_linker_gen"].shape[0]): #for each sample
            mask_position_linker_gen[i, 0:out["num_linker_gen_nodes"][i]] = 1 #go to that row and from column 0 to "num atoms in that sample" put 1, leave the rest of the row with 0
        
        mask_position_fragment = torch.zeros(size=(out["position_fragment"].shape[0], out["position_fragment"].shape[1])) #tensor full of 0 with shape [number of samples in dataset, max num of fragment atoms]
        for i in range(out["position_fragment"].shape[0]): #for each sample
            mask_position_fragment[i, 0:out["num_fragment_nodes"][i]] = 1 #go to that row and from column 0 to column "num atoms in that sample" put 1, leave the rest of the row with 0

        mask_position_protein = torch.zeros(size=(out["position_protein"].shape[0], out["position_protein"].shape[1])) #tensor full of 0 with shape [number of samples in dataset, max num of fragment atoms]
        for i in range(out["position_protein"].shape[0]): #for each sample
            mask_position_protein[i, 0:out["num_protein_nodes"][i]] = 1 #go to that row and from column 0 to column "num atoms in that sample" put 1, leave the rest of the row with 0

        out["mask_position_linker_gen"], out["mask_position_fragment"], out["mask_position_protein"] = mask_position_linker_gen, mask_position_fragment, mask_position_protein #add mask tensor to out dict
        del mask_position_linker_gen, mask_position_fragment, mask_position_protein

        ## category masks
        mask_category_linker_gen = torch.zeros_like(out["category_linker_gen"]) #create a tensor full of 0 of shape [num instances in dataset, max num linker nodes]
        for i in range(mask_category_linker_gen.shape[0]): #for each sample
            mask_category_linker_gen[i, 0:out["num_linker_gen_nodes"][i]] = 1 #go to that row and from the beginning to the number of atoms in that sample put 1, leave the rest with 0
        
        mask_category_fragment = torch.zeros_like(out["category_fragment"]) #create a tensor full of 0 of shape [num instances in dataset, max num linker nodes]
        for i in range(mask_category_fragment.shape[0]): #for each sample
            mask_category_fragment[i, 0:out["num_fragment_nodes"][i]] = 1 #go to that row and from the beginning to the number of atoms in that sample put 1, leave the rest with 0

        mask_category_protein = torch.zeros_like(out["category_protein"]) #create a tensor full of 0 of shape [num instances in dataset, max num linker nodes]
        for i in range(mask_category_protein.shape[0]): #for each sample
            mask_category_protein[i, 0:out["num_protein_nodes"][i]] = 1 #go to that row and from the beginning to the number of atoms in that sample put 1, leave the rest with 0

        out["mask_category_linker_gen"], out["mask_category_fragment"], out["mask_category_protein"] = mask_category_linker_gen, mask_category_fragment, mask_category_protein
        del mask_category_linker_gen, mask_category_fragment, mask_category_protein

        ## edge masks
        inf_mask_linker_gen = torch.isinf(out["linker_edge_list"]).all(dim=2)  # Returns a boolean tensor of the same shape as protein_chopped_edge_list indicating where the inf values are located
        mask_linker_gen_edge_list = torch.ones(size=(out["linker_edge_list"].shape[0], out["linker_edge_list"].shape[1])) # Create a matrix filled with ones with shape [number of samples, max number of edges]
        mask_linker_gen_edge_list[inf_mask_linker_gen] = 0 # Set the corresponding positions to 0 where the mask is True

        inf_mask_fragment = torch.isinf(out["fragment_edge_list"]).all(dim=2)  # Returns a boolean tensor of the same shape as protein_chopped_edge_list indicating where the inf values are located
        mask_fragment_edge_list = torch.ones(size=(out["fragment_edge_list"].shape[0], out["fragment_edge_list"].shape[1])) # Create a matrix filled with ones with shape [number of samples, max number of edges]
        mask_fragment_edge_list[inf_mask_fragment] = 0 # Set the corresponding positions to 0 where the mask is True


        inf_mask_protein = torch.isinf(out["protein_chopped_edge_list"]).all(dim=2)  # Returns a boolean tensor of the same shape as protein_chopped_edge_list indicating where the inf values are located
        mask_protein_edge_list = torch.ones(size=(out["protein_chopped_edge_list"].shape[0], out["protein_chopped_edge_list"].shape[1])) # Create a matrix filled with ones with shape [number of samples, max number of edges]
        mask_protein_edge_list[inf_mask_protein] = 0 # Set the corresponding positions to 0 where the mask is True

        out["mask_edge_list_linker_gen"], out["mask_edge_list_fragment"], out["mask_edge_list_protein"] = mask_linker_gen_edge_list, mask_fragment_edge_list, mask_protein_edge_list
        del mask_linker_gen_edge_list, mask_fragment_edge_list, mask_protein_edge_list

        return out #return the dictionary


    def get_dataloader(self):
        """
        For each subset (train, valid or test) and the associated instances, perform a preprocessing step using a custom collate fn, then return the three dataloaders.

        For the chosen subset return the dataloader.
        
        Returns:
        --------

        Dataloader


        {"train": Dataloader for train,
        "valid": Dataloader for validation,
        "test": Dataloader for test}
        """
        if self.which_dataset == "train":
            shuffle=True
        else:
            shuffle=False

        loaded_dataset = self.load_subset()

        return  DataLoader(loaded_dataset, self.dataset_config.batch_size, shuffle=shuffle, collate_fn=self.custom_collate)
                