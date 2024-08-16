from dataclasses import dataclass
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from torch.utils.data import DataLoader
import torch
from typing import Literal

#### make configuration class for LP dataset
@dataclass
class LPConfig:
    batch_size : int = 32 #batch size
    num_workers : int = 6 #number of subprocesses used to load the data (0 = sequantial data loading: may create a bottleneck)

    max_num_protein_nodes : int = 500 #max number of nodes a protein can have to pass the filter in the collate function of the dataloader
    accept_variable_bs : bool = False #if true, the filter that discard instances where protein has more than max_num_protein_nodes nodes is done in the collate funciton (which can result in batches with different number of samples and possibly empty batches). If false, the filtering is done before so that the dataloader receives allready filtered instances

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


#### make dataloader class for LP dataset
class LPDataloader(MarkovBridgeDataloader):

    def __init__(self):#, which_dataset:Literal["train", "valid", "test"]):
        #check that the dataset requested is correct, raise ValueError instead
        #if which_dataset not in ["train", "valid", "test"]:
        #    raise ValueError(f"Invalid value for which_dataset: {which_dataset}. Must be one of ['train', 'valid', 'test']")
        
        #self.which_dataset = which_dataset #which subset you want (train, valid or test)
        self.dataset_config = LPConfig #dataset configuration
        

    def _load_subsets(self): 
        """
        Loads train, validation and test set .pt files: those are list of dictionaries where each dictioary contains information for an instance.
        Obtains the entire list or a part of it according to the desred number of instances to retrieve from that set, 
        then returns a list with the selected instances (all or just some).

        Returns:
        --------
       {"train": selected_train,
                "valid": selected_valid,
                "test": selected_test}
        """
        #if self.which_dataset == "train":
        #    train = torch.load(self.dataset_config.train_path)
        #    selected_dataset = train if self.dataset_config.num_pts_train == -1 else train[:self.dataset_config.num_pts_train] #select training instances to be returned in batch by the dataloader (all or a certain number specified in num_pts_train)
        #elif self.which_dataset == "valid":
        #    valid = torch.load(self.dataset_config.valid_path)
        #    selected_dataset = valid if self.dataset_config.num_pts_valid == -1 else valid[:self.dataset_config.num_pts_valid] #select validation instances
        #elif self.which_dataset == "test":
        #    test = torch.load(self.dataset_config.test_path)
        #    selected_dataset = test if self.dataset_config.num_pts_test == -1 else test[:self.dataset_config.num_pts_test] #select test instances

        ### load the entire dataset
        train = torch.load(self.dataset_config.train_path)
        valid = torch.load(self.dataset_config.valid_path)
        test = torch.load(self.dataset_config.test_path)

        ### filter for the number of protein nodes
        if self.dataset_config.accept_variable_bs == False: #by default this is executed
            train = [instance for instance in train if instance["num_protein_nodes"]<=self.dataset_config.max_num_protein_nodes] #for each element in the list of dictionaries where every dictionary represent an instance, take only those where the number of nodes is below 500
            valid = [instance for instance in valid if instance["num_protein_nodes"]<=self.dataset_config.max_num_protein_nodes]
            test = [instance for instance in test if instance["num_protein_nodes"]<=self.dataset_config.max_num_protein_nodes]
            print(f"[INFO]\nNumber of training instances after max_num_protein_nodes filter at {self.dataset_config.max_num_protein_nodes}: {len(train)}\nNumber of validation instances after max_num_protein_nodes filter at {self.dataset_config.max_num_protein_nodes}: {len(valid)}\nNumber of test instances after max_num_protein_nodes filter at {self.dataset_config.max_num_protein_nodes}: {len(test)}\n")

        ### select the number of instances tot ake from each filtered dataset
        selected_train = train if self.dataset_config.num_pts_train == -1 else train[:self.dataset_config.num_pts_train] #select training instances to be returned in batch by the dataloader (all or a certain number specified in num_pts_train)
        selected_valid = valid if self.dataset_config.num_pts_valid == -1 else valid[:self.dataset_config.num_pts_valid] #select validation instances
        selected_test = test if self.dataset_config.num_pts_test == -1 else test[:self.dataset_config.num_pts_test] #select test instances

        return {"train": selected_train,
                "valid": selected_valid,
                "test": selected_test}



    def _custom_collate(self, data):
        """
        Custom collate function. 

        It takes as input a batch of data organized as returned by the load_subset function. It rearranges information, performs padding and create masks after padding. 
        Then, returns a dictionary with unique keys containing information and padding masks for the instances in the batch.

        Notes:
        ----------
        data is a list of dictionaries like [{"mol_number":1, "mol_pos":torch.tensor with positions of molecule 1, "mol_onehot":torch.tensor with onehot of molecule one}, {same for mol2..}]:
        it contains a batch of instances from the selected subset (train, valid or test returned by load_subset function).

        custom_collate first creates the out dictionary to be like {"mol_number": torch.tensor of all molecules number, "mol_pos":torch.tensor of all molecule atoms positions} ...,
        then padd all values specified in config KEYS_TO_PAD, and finally create the masks for padded things.
        """
        if self.dataset_config.accept_variable_bs: #by default is false so this "if" is not executed
            #### filter out instances where the number of protein nosed is > 500 (500 is a parameter that can be set from the dataset config class)
            data = [instance for instance in data if instance["num_protein_nodes"]<=self.dataset_config.max_num_protein_nodes] #for each element in the list of dictionaries where every dictionary represent an instance, take only those where the number of nodes is below 500

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

    def _get_shuffle_value(self, split):
        # Fetch the shuffle value based on the split key
        if split == 'train':
            return self.dataset_config.shuffle_train
        elif split == 'valid':
            return self.dataset_config.shuffle_valid
        elif split == 'test':
            return self.dataset_config.shuffle_test
        else:
            raise ValueError(f"Unknown split: {split}. Mus be one of ['train', 'valid', 'test']")

    def _get_dataloaders(self):
        """
        For the chosen subset ("train", "valid" or "test" returned by load_subset function) returns the corresponding dataloader.
        """
        #set shuffle bool value according to self.which_subset
        #if self.which_dataset == "train":
        #    shuffle = self.dataset_config.shuffle_train 
        #elif self.which_dataset == "valid":
        #    shuffle = self.dataset_config.shuffle_valid
        #elif self.which_dataset == "test":
        #    shuffle = self.dataset_config.shuffle_test

        #loaded_datasets = self.load_subsets() #load the desired dataset

        #return  DataLoader(dataset=loaded_dataset, batch_size=self.dataset_config.batch_size, shuffle=shuffle, num_workers=self.dataset_config.num_workers, collate_fn=self.custom_collate) #return the dataloader
        return {split: DataLoader(dataset=dataset, batch_size=self.dataset_config.batch_size, shuffle=self._get_shuffle_value(split), num_workers=self.dataset_config.num_workers, collate_fn=self._custom_collate) 
                for split, dataset in self._load_subsets().items()}    
    
    def train(self):
        return self._get_dataloaders()["train"]
    
    def valid(self):
        return self._get_dataloaders()["valid"]
    
    def test(self):
        return self._get_dataloaders()["test"]
    
    def get_databatch_keys(self, check=False):
        if check:
            ### retrieve the first batch from eanch of the three dataloader
            first_batches = {}
            for key, loader in self._get_dataloaders().items():
                first_batch = next(iter(loader))# Get the first batch
                first_batches[key] = first_batch #put it in the dictionay associating it with the corret dataset name (train, valid or test)
            ### retrieve keys in each of the three batches and check that they corrspond
            batch_keys = {key: list(batch.keys()) for key, batch in first_batches.items()}
            assert all(keys == batch_keys[list(batch_keys.keys())[0]] for keys in batch_keys.values()), "Datasets must have same set of keys!" # Ensure all keys are the same
        else: #(default)
            #### load only first batch of test set, do not check for key correspondence but instead just return the keys in the first batch of test set (do only if you are sure that keys correspond)
            first_batches = {}
            for key, loader in self._get_dataloaders().items():
                if key=="test":
                    first_batch = next(iter(loader))# Get the first batch
                    first_batches[key] = first_batch #put it in the dictionay associating it with the corret dataset name (train, valid or test)
        return list(first_batches["test"].keys()) #batch_keys #[first_batches["test"].keys()]
