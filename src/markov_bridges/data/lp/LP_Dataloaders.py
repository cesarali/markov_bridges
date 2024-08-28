from dataclasses import dataclass, asdict
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from torch.utils.data import DataLoader
import torch
from markov_bridges.configs.config_classes.data.molecules_configs import LPConfig
from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMG_LPConfig
from typing import Literal

#### make dataloader class for LP dataset
class LPDataloader(MarkovBridgeDataloader):

    def __init__(self, config:EDMG_LPConfig): ##by default take the LPConfig values, but if required you can create an instance of LPConfig, change some field's values and pass that to the LPDataloader instance
        #self.dataset_config = LPConfig() #dataset configuration
        self.dataset_config = config.data
        self.dataset_info = self._get_dataset_info()
        

    def _load_subsets(self): 
        """
        Loads train, validation and test set .pt files: those are list of dictionaries where each dictioary contains information for an instance.

        If accept_variable_bs == False (default recommended option) it filter out instances where the protein has more than 500 nodes (>500 out; <=500 in). Else, this filtering procedure is not done now but later (in the collate function)

        Then, it obtains the entire resulting list or a part of it according to the desired number of instances to retrieve from that set (as specified in num_pts_train, num_pts_valid, num_pts_test).

        If padding_dependence == "dataset" it pads up to the max value in each dataset, create the masks for padding and returns the padded datasets (in this case, no custom collate function will be used in the later call of the dataloader)

        Fnally returns a dictionary of 3 lists with the selected instances.

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

        ### filter for the number of protein, fragment and linker nodes if accept_variable_bs == False (mandatory; default value)
        if self.dataset_config.accept_variable_bs == False: #by default this is executed
            #filter for protein nodes
            train = [instance for instance in train if instance["num_protein_nodes"]<=self.dataset_config.max_num_protein_nodes] #for each element in the list of dictionaries where every dictionary represent an instance, take only those where the number of nodes is below 500
            train = [instance for instance in train if instance["num_protein_nodes"]>=self.dataset_config.min_num_protein_nodes]
            valid = [instance for instance in valid if instance["num_protein_nodes"]<=self.dataset_config.max_num_protein_nodes]
            valid = [instance for instance in valid if instance["num_protein_nodes"]>=self.dataset_config.min_num_protein_nodes]
            test = [instance for instance in test if instance["num_protein_nodes"]<=self.dataset_config.max_num_protein_nodes]
            test = [instance for instance in test if instance["num_protein_nodes"]>=self.dataset_config.min_num_protein_nodes]
            #filter for linker nodes
            train = [instance for instance in train if instance["num_linker_gen_nodes"]<=self.dataset_config.max_num_linker_nodes] #for each element in the list of dictionaries where every dictionary represent an instance, take only those where the number of nodes is below 500
            train = [instance for instance in train if instance["num_linker_gen_nodes"]>=self.dataset_config.min_num_linker_nodes]
            valid = [instance for instance in valid if instance["num_linker_gen_nodes"]<=self.dataset_config.max_num_linker_nodes]
            valid = [instance for instance in valid if instance["num_linker_gen_nodes"]>=self.dataset_config.min_num_linker_nodes]
            test = [instance for instance in test if instance["num_linker_gen_nodes"]<=self.dataset_config.max_num_linker_nodes]
            test = [instance for instance in test if instance["num_linker_gen_nodes"]>=self.dataset_config.min_num_linker_nodes]
            #filter for fragment atoms
            train = [instance for instance in train if instance["num_fragment_nodes"]<=self.dataset_config.max_num_fragment_nodes] #for each element in the list of dictionaries where every dictionary represent an instance, take only those where the number of nodes is below 500
            train = [instance for instance in train if instance["num_fragment_nodes"]>=self.dataset_config.min_num_fragment_nodes]
            valid = [instance for instance in valid if instance["num_fragment_nodes"]<=self.dataset_config.max_num_fragment_nodes]
            valid = [instance for instance in valid if instance["num_fragment_nodes"]>=self.dataset_config.min_num_fragment_nodes]
            test = [instance for instance in test if instance["num_fragment_nodes"]<=self.dataset_config.max_num_fragment_nodes]
            test = [instance for instance in test if instance["num_fragment_nodes"]>=self.dataset_config.min_num_fragment_nodes]

            #print(f"[INFO]\nNumber of training instances after max_num_protein_nodes filter at {self.dataset_config.max_num_protein_nodes}: {len(train)}\nNumber of validation instances after max_num_protein_nodes filter at {self.dataset_config.max_num_protein_nodes}: {len(valid)}\nNumber of test instances after max_num_protein_nodes filter at {self.dataset_config.max_num_protein_nodes}: {len(test)}\n")

        ### select the number of instances to take from each filtered dataset
        selected_train = train if self.dataset_config.num_pts_train == -1 else train[:self.dataset_config.num_pts_train] #select training instances to be returned in batch by the dataloader (all or a certain number specified in num_pts_train)
        selected_valid = valid if self.dataset_config.num_pts_valid == -1 else valid[:self.dataset_config.num_pts_valid] #select validation instances
        selected_test = test if self.dataset_config.num_pts_test == -1 else test[:self.dataset_config.num_pts_test] #select test instances

        ### if padding_dependence = "dataset" pad up to the max vaues in each dataset, create the masks after padding and return the padded datasets
        if self.dataset_config.padding_dependence == "dataset":
            for label, dataset in {"train":selected_train, "valid":selected_valid, "test":selected_test}.items(): #foreach of the three datasets
                #### restyle data in the form of dictionay with unique keys, where to each key there are associated values of all instances
                out = {} #intialize out dictionary: this dictionary will contain unique keys and all the subset values
                for sample in dataset: #for each sample in the subset
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

                #### substitute padded inf with 0 values
                for k, v in out.items():
                    if k in self.dataset_config.KEYS_TO_PAD: #for elements that have been padded
                        out[k] = torch.nan_to_num(v, posinf=0) #substitute all inf with 0

                ### revert out dictionary to be again a list of dictionaries where each dictionary represent one instance
                dataset_list = []
                num_samples = len(next(iter(out.values())))
                for i in range(num_samples):
                    sample = {}
                    for k, v_list in out.items():
                        sample[k] = v_list[i]
                    dataset_list.append(sample)

                ### according to the label, change the value of selected_train, selected_val or selected_test
                if label == "train":
                    selected_train = dataset_list
                elif label == "valid":
                    selected_valid = dataset_list
                elif label == "test":
                    selected_test = dataset_list

        return {"train": selected_train,
                "valid": selected_valid,
                "test": selected_test}



    def _custom_collate(self, data):
        """
        Custom collate function. 

        [IMPO] It paddes data per batch. This means that if n the first batch max_num_nodes = 10 and in the second batch max_num_nodes = 15, the first batch will be padded up to 10 and the second up to 15.
               Use this collate function only if the padding_dependence value in the config file is set to "batch" 

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
            raise NotImplementedError("bool True for accept_variable_bs has not been implemented. Set this variable to False.")
            #OLD: accept True and apply here the filter only for the max num of protein nodes. Now deprecated: True raises NotImplementedError.
            #### filter out instances where the number of protein nosed is > 500 (500 is a parameter that can be set from the dataset config class)
            #print(f"[INFO] filter for max_num_protein nodes at {self.dataset_config.max_num_protein_nodes} is being applied in the collate function. This may cause different batch sizes. Is recommended to use accept_variable_bs=False")
            #data = [instance for instance in data if instance["num_protein_nodes"]<=self.dataset_config.max_num_protein_nodes] #for each element in the list of dictionaries where every dictionary represent an instance, take only those where the number of nodes is below 500

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
        mask_linker_gen_edge_list[inf_mask_linker_gen] = 0 # Set the corresponding positions to 0 where the mask is True (i.e., where there are inf values)

        inf_mask_fragment = torch.isinf(out["fragment_edge_list"]).all(dim=2)  # Returns a boolean tensor of the same shape as protein_chopped_edge_list indicating where the inf values are located
        mask_fragment_edge_list = torch.ones(size=(out["fragment_edge_list"].shape[0], out["fragment_edge_list"].shape[1])) # Create a matrix filled with ones with shape [number of samples, max number of edges]
        mask_fragment_edge_list[inf_mask_fragment] = 0 # Set the corresponding positions to 0 where the mask is True


        inf_mask_protein = torch.isinf(out["protein_chopped_edge_list"]).all(dim=2)  # Returns a boolean tensor of the same shape as protein_chopped_edge_list indicating where the inf values are located
        mask_protein_edge_list = torch.ones(size=(out["protein_chopped_edge_list"].shape[0], out["protein_chopped_edge_list"].shape[1])) # Create a matrix filled with ones with shape [number of samples, max number of edges]
        mask_protein_edge_list[inf_mask_protein] = 0 # Set the corresponding positions to 0 where the mask is True

        out["mask_edge_list_linker_gen"], out["mask_edge_list_fragment"], out["mask_edge_list_protein"] = mask_linker_gen_edge_list, mask_fragment_edge_list, mask_protein_edge_list
        del mask_linker_gen_edge_list, mask_fragment_edge_list, mask_protein_edge_list

        #### substitute padded inf with 0 values
        for k, v in out.items():
            if k in self.dataset_config.KEYS_TO_PAD: #for elements that have been padded
                out[k] = torch.nan_to_num(v, posinf=0) #substitute all inf with 0

        return out #return the dictionary

    def _get_shuffle_value(self, split):
        """
        Fetch the shuffle value based on the split key: train has shuffle True, valid and test have shuffle False.
        This funciton return the bool value for shuffle according to is we are asking for train, valid or test
        """
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
        For each subset ("train", "valid" and "test" returned by load_subset function) returns the corresponding dataloader.

        Returns:
        ---------
        {"train": dataloader for train,
                "valid": dataloader for validation,
                "test": dataloader for test}
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
        
        ### according to the padding_dependence flag, select if the custom collate function is to be used or not. This function should be used only in case of padding_dependence="batch" because it returns padding and masks per batch
        if self.dataset_config.padding_dependence=="batch":
            collate_to_use = self._custom_collate
        else:
            collate_to_use = None
        
        return {split: DataLoader(dataset=dataset, batch_size=self.dataset_config.batch_size, shuffle=self._get_shuffle_value(split), num_workers=self.dataset_config.num_workers, collate_fn=collate_to_use) 
                for split, dataset in self._load_subsets().items()}    
    
    def train(self): #returns just the train dataloader 
        return self._get_dataloaders()["train"]
    
    def valid(self): #return just the validation dataloader
        return self._get_dataloaders()["valid"]
    
    def test(self): #return just the test dataloader
        return self._get_dataloaders()["test"]
    
    def get_databatch_keys(self, check=False):
        """
        Get the keys that we find in the batch
        """
        if check:
            ### retrieve the first batch from each of the three dataloader
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
        return list(first_batches["test"].keys()) #batch_keys 
    

    def _get_hist_dicts(self, dataset):
        """
        Function to retrieve from a loaded and eventually filtered dataset (train, test or valid) the number of linker nodes, fragment nodes and protein nodes

        Returns:
        ---------
        linker_atoms_dict, fragment_atoms_dict, protein_atoms_dict : dicts with the counts of how many instances have that number of linker nodes, fragment nodes and protein nodes.
        es. linker_atoms_dict = {4: 44, 5: 35, 6:26} means that 44 instances have 4 linker nodes, 35 instances have 5 linker nodes and 26 instances have 6 linker nodes. 
        """
        num_linker_nodes, num_fragment_nodes, num_protein_nodes = [], [], [] #initialize 3 empty lists
        for instance in dataset: #for each instance
            num_linker_nodes.append(instance["num_linker_gen_nodes"]), num_fragment_nodes.append(instance["num_fragment_nodes"]), num_protein_nodes.append(instance["num_protein_nodes"]) #extract value and apend it to the correct list
        num_linker_nodes , num_fragment_nodes, num_protein_nodes = torch.tensor(num_linker_nodes), torch.tensor(num_fragment_nodes), torch.tensor(num_protein_nodes) #convert list to torch tensors for counting operation
        unique_values_linker, count_linker = torch.unique(num_linker_nodes, sorted=True, return_counts=True) #get unique values and their counts for linker
        unique_values_fragment, count_fragment = torch.unique(num_fragment_nodes, sorted=True, return_counts=True) #get unique values and their count for fragment
        unique_values_protein, count_protein = torch.unique(num_protein_nodes, sorted=True, return_counts=True) #get unique values and their count for protein
        linker_atoms_dict = {int(value):int(count) for value, count in zip(unique_values_linker, count_linker)} #create dict for linker
        fragment_atoms_dict = {int(value):int(count) for value, count in zip(unique_values_fragment, count_fragment)} #create dict for fragment
        protein_atoms_dict = {int(value):int(count) for value, count in zip(unique_values_protein, count_protein)} #create dict for fragment
        return linker_atoms_dict, fragment_atoms_dict, protein_atoms_dict


    def _get_dataset_info(self):

        info = {} #dictionary to return which contains inforation

        ### add to the info dictionary the config value used
        info["General_Configuration"] = asdict(self.dataset_config)

        ### load the entire 3 datasets
        train = torch.load(self.dataset_config.train_path)
        valid = torch.load(self.dataset_config.valid_path)
        test = torch.load(self.dataset_config.test_path)

        ### filter for the number of protein nodes if accept_variable_bs == False (recommended; default value)
        if self.dataset_config.accept_variable_bs == False: #by default this is executed
            train = [instance for instance in train if instance["num_protein_nodes"]<=self.dataset_config.max_num_protein_nodes] #for each element in the list of dictionaries where every dictionary represent an instance, take only those where the number of nodes is below 500
            valid = [instance for instance in valid if instance["num_protein_nodes"]<=self.dataset_config.max_num_protein_nodes]
            test = [instance for instance in test if instance["num_protein_nodes"]<=self.dataset_config.max_num_protein_nodes]
            #print(f"[INFO]\nNumber of training instances after max_num_protein_nodes filter at {self.dataset_config.max_num_protein_nodes}: {len(train)}\nNumber of validation instances after max_num_protein_nodes filter at {self.dataset_config.max_num_protein_nodes}: {len(valid)}\nNumber of test instances after max_num_protein_nodes filter at {self.dataset_config.max_num_protein_nodes}: {len(test)}\n")

        ### select the number of instances to take from each filtered dataset
        selected_train = train if self.dataset_config.num_pts_train == -1 else train[:self.dataset_config.num_pts_train] #select training instances to be returned in batch by the dataloader (all or a certain number specified in num_pts_train)
        selected_valid = valid if self.dataset_config.num_pts_valid == -1 else valid[:self.dataset_config.num_pts_valid] #select validation instances
        selected_test = test if self.dataset_config.num_pts_test == -1 else test[:self.dataset_config.num_pts_test] #select test instances

        info["train"], info["valid"], info["test"] = {}, {}, {} #initialize 3 dictioanaries, one for each dataset

        ### number of instances in each dataset after filtering and num_pts selection
        info["train"]["number_of_instances"], info["valid"]["number_of_instances"], info["test"]["number_of_instances"] = len(selected_train), len(selected_valid), len(selected_test)

        ### histogram values for number of linker_gen_nodes in each dataset
        info["train"]["count_linker_gen_nodes"], info["train"]["count_fragment_nodes"], info["train"]["count_protein_nodes"] = self._get_hist_dicts(selected_train)
        info["valid"]["count_linker_gen_nodes"], info["valid"]["count_fragment_nodes"], info["valid"]["count_protein_nodes"] = self._get_hist_dicts(selected_valid)
        info["test"]["count_linker_gen_nodes"], info["test"]["count_fragment_nodes"], info["test"]["count_protein_nodes"] = self._get_hist_dicts(selected_test)

        return info

