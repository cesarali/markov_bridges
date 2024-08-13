

import dill, os, torch
from tqdm.notebook import tqdm

train_in_path = "/home/piazza/markov_bridges/LP_Data/ReducedDiffusionDataset/Train" #our training set (input) path
val_in_path = "/home/piazza/markov_bridges/LP_Data/ReducedDiffusionDataset/Validation" #our validation set (input) path
test_in_path = "/home/piazza/markov_bridges/LP_Data/ReducedDiffusionDataset/Test" #our test set (input) path


## Train set 
print("Train conversion")
out_list , id_counter = [] , 0 #empty list where to append all train set molecule's dictionaries, counter for the number of training molecules
filename_list = os.listdir(train_in_path)
filename_list.sort() #sort pkl files
for file_name in tqdm(filename_list): #for each train .pkl file
    file = dill.load(open(f"{train_in_path}/{file_name}", "rb"))
    for instance in file: #for each graph stored in the pkl file
        out = {} #create an empty dictionary which represent this graph; this dictionary will be appended to the out_train_list
        out["uuid"] = id_counter #graph number
        id_counter += 1 #increment sample number
        out["name"] = instance["instanceFilename"] #track filename (contains also the rcsb id as prefix)
        out["positions"] = instance["x"][:,:3] #nodes coordinates are the first 3 columns
        out["one_hot"] = instance["x"][:, 3:-1] #one hot encoding of atom type 
        out["linker_mask"] = instance["x"][:, -1] #last column is linker mask : 1 if the node belongs to the linker and needs to be diffused, 0 if it is a fragment or sphere node which has to remain fixed 
        out["context_mask"] = 1-out["linker_mask"] #opposite of linker mask
        out_list.append(out) #append the dictionary to the list
        del instance, out #free mem
    del file, file_name #free mem
torch.save(out_list, "/home/piazza/markov_bridges/LP_Data/RestyledReducedDiffusionDataset/train.pt") #save 
del out_list , id_counter , filename_list #free mem

## Validation set
print("Validation conversion")
out_list , id_counter = [] , 0 #empty list where to append all val set molecule's dictionaries, counter for the number of training molecules
filename_list = os.listdir(val_in_path)
filename_list.sort() #sort pkl files
for file_name in tqdm(filename_list): #for each val .pkl file
    file = dill.load(open(f"{val_in_path}/{file_name}", "rb"))
    for instance in file: #for each graph stored in the pkl file
        out = {} #create an empty dictionary which represent this graph; this dictionary will be appended to the out_train_list
        out["uuid"] = id_counter #graph number
        id_counter += 1 #increment sample number
        out["name"] = instance["instanceFilename"] #track filename (contains also the rcsb id as prefix)
        out["positions"] = instance["x"][:,:3] #nodes coordinates are the first 3 columns
        out["one_hot"] = instance["x"][:, 3:-1] #one hot encoding of atom type 
        out["linker_mask"] = instance["x"][:, -1] #last column is linker mask : 1 if the node belongs to the linker and needs to be diffused, 0 if it is a fragment or sphere node which has to remain fixed (context)
        out["context_mask"] = 1-out["linker_mask"] #opposite of linker mask
        out_list.append(out) #append the dictionary to the list
        del instance, out #free mem
    del file, file_name #free mem
torch.save(out_list, "/home/piazza/markov_bridges/LP_Data/RestyledReducedDiffusionDataset/validation.pt") #save 
del out_list , id_counter, filename_list #free mem

## Test set
print("Test conversion")
out_list , id_counter = [] , 0 #empty list where to append all val set molecule's dictionaries, counter for the number of training molecules
filename_list = os.listdir(test_in_path)
filename_list.sort() #sort pkl files
for file_name in tqdm(filename_list): #for each test .pkl file
    file = dill.load(open(f"{test_in_path}/{file_name}", "rb"))
    for instance in file: #for each graph stored in the pkl file
        out = {} #create an empty dictionary which represent this graph; this dictionary will be appended to the out_train_list
        out["uuid"] = id_counter #graph number
        id_counter += 1 #increment sample number
        out["name"] = instance["instanceFilename"] #track filename (contains also the rcsb id as prefix)
        out["positions"] = instance["x"][:,:3] #nodes coordinates are the first 3 columns
        out["one_hot"] = instance["x"][:, 3:-1] #one hot encoding of atom type 
        out["linker_mask"] = instance["x"][:, -1] #last column is linker mask : 1 if the node belongs to the linker and needs to be diffused, 0 if it is a fragment or sphere node which has to remain fixed (context)
        out["context_mask"] = 1-out["linker_mask"] #opposite of linker mask
        out_list.append(out) #append the dictionary to the list
        del instance, out #free mem
    del file, file_name #free mem
torch.save(out_list, "/home/piazza/markov_bridges/LP_Data/RestyledReducedDiffusionDataset/test.pt") #save 
del out_list , id_counter , filename_list #free mem

print("Conversion finished")
