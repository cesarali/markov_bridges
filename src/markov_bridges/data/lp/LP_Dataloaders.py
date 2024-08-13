# dataloader for lp dataset


import torch
from torch.utils.data import Dataset, DataLoader

#define custom class for the dataloader
class LPDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

    
def _collate(self, dataset):
    """
    Custom collate function. 
    It takes as input a dataset (can be train, validation or test set), and rearrange informations.

    dataset is a list of dictionaries like [{"mol_number":1, "mol_pos":torch.tensor with positions of molecule 1, "mol_onehot":torch.tensor with onehot of molecule one}, {same for mol2..}]
    and returns a dictionary like {"mol_number": torch.tensor of all molecules number, "mol_pos":torch.tensor of all molecule atoms positions} ...
    """
    out = {} #intialize out dictionary
    for sample in dataset: #for each sample in the dataset
        for key, value in sample.items(): #iterate through keys
            out.setdefault(key, []).append(value) #set keys to out dictionary if not yet existing, then append values to the key
    
    
    #padding

    
def get_dataloaders(self, train_path, val_path, test_path, batch_size):
    train, val, test = torch.load(train_path), torch.load(val_path), torch.load(test_path) #load .pt files -> obtain list of dictionaries where every dictionary is for a sample
