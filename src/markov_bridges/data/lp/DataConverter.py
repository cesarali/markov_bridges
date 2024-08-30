

import dill, os, torch, math

train_in_path = "/home/piazza/markov_bridges/LP_Data/ReducedDiffusionDataset/Train" #our training set (input) path
val_in_path = "/home/piazza/markov_bridges/LP_Data/ReducedDiffusionDataset/Validation" #our validation set (input) path
test_in_path = "/home/piazza/markov_bridges/LP_Data/ReducedDiffusionDataset/Test" #our test set (input) path



print("Train conversion")
out_list , id_counter = [] , 0 #empty list where to append all train set molecule's dictionaries, counter for the number of training molecules
filename_list = os.listdir(train_in_path)
filename_list.sort() #sort pkl files
for file_name in filename_list: #for each train .pkl file
    print("processing pkl")
    file = dill.load(open(f"{train_in_path}/{file_name}", "rb"))
    for instance in file: #for each graph stored in the pkl file
        out = {} #create an empty dictionary which represent this instance; this dictionary will be appended to the out_list
        out["uuid"] = id_counter #instance number number
        id_counter += 1 #increment number
        out["name"] = instance["instanceFilename"] #track filename (contains also the rcsb id as prefix)
        
        #### retrieve coordinates, onehot encoding and linker/context flag
        coordinates = instance["x"][:,:3] #nodes coordinates are the first 3 columns. THOSE ARE ALL NODES COORDINATES
        one_hot = instance["x"][:, 3:-1] #one hot encoding of atom type. THIS IS THE ONEHOT FOR ALL NODES
        linker_flag = instance["x"][:, -1] #binary flag which discriminate linker nodes (1) from context nodex (0). THIS IS THE LINKER FLAG FOR ALL NODES
        context_flag = 1-linker_flag #binary flag which discriminate linker nodes (0) from context nodex (1). THIS IS THE CONTEXT FLAG FOR ALL NODES

        #### covert the onehot encoding into a single number (es [0,0,0,1,0] -> 3)
        category = torch.argmax(one_hot, dim=1) #Atom type retrieved from one hot encoding. atom type is 0 for exclusion spheres. THIS IS THE CATEGORY FOR ALL NODES
        
        #### get positions and category for linker atoms by discriminating linker atoms from context atoms using the linker flag
        positions_linker = coordinates[linker_flag.bool()] #use the binary flag to identify linker nodes, then retrieve only their positions
        category_linker = category[linker_flag.bool()] #use the binary flag to identify linker nodes, then retrieve only their category
        
        #### get positions and category for context atoms (context = fragment & protein) by using the context flag
        positions_context = coordinates[context_flag.bool()] #use binary flag to identify context nodes, ten retrieve only their position
        category_context = category[context_flag.bool()] #use binary flag to identify context nodes, then retireve only their category

        #### within the context, discriminate fragment from protein nodes using the category (0 is for protein, all other categories are fragment atoms)
        mask_category_0_context = (category_context == 0) #boolean mask that identify which context nodes belong to category 0 and which not
        positions_protein = positions_context[mask_category_0_context] #get position of only protein nodes
        category_protein = category_context[mask_category_0_context] #get the category for protein nodes (this wille full of 0)
        positions_fragment = positions_context[~mask_category_0_context] #get position of fragment nodes by the negation of the protein mask nodes
        category_fragment = category_context[~mask_category_0_context] #get categories of fragment nodes
        
        #### retrieve the number of linker nodes, fragment nodes and protein nodes
        num_linker_nodes, num_fragment_nodes, num_protein_nodes = category_linker.shape[0], category_fragment.shape[0], category_protein.shape[0]

        print("getting protein edge list")
        #### for the protein part, build the edge list by adding an edge only if the node distance is <= 2.5 A. the edge list for 1 protein has shape [2, num_existing_edges*2], the *2 is because the graph is undirected
        sender_protein_nodes, receiver_protein_nodes = [], [] #empty list where to store indexes of sender and receiver node of an edge in the protein part
        for i, i_coord in enumerate(positions_protein): #for each protein node and its coordinates
            for j, j_coord in enumerate(positions_protein): #with all other protein nodes (included itself) and their coordinates 
                dist = math.dist(i_coord, j_coord) #calculate euclidean distance between nodes
                if dist <= 4: #if the distance is below the threshold then the edge exist (in this way also self loop exists)
                    sender_protein_nodes.append(i) #i is sender (in this double iteration i will later be the receiver and  j the sender)
                    receiver_protein_nodes.append(j) #j is receiver
                del j, j_coord, dist #free mem in loop
            del i, i_coord #free mem in loop
        protein_chopped_edge_list = [torch.LongTensor(sender_protein_nodes), torch.LongTensor(receiver_protein_nodes)] #edge list after chopping edges above the threshold

        print("getting linker edge list")
        #### for the linker part, build the edge list considering that the linker is FC
        sender_linker_nodes, receiver_linker_nodes = [], []
        for i, i_coord in enumerate(positions_linker):
            for j, j_coord in enumerate(positions_linker):
                sender_linker_nodes.append(i)
                receiver_linker_nodes.append(j)
                del j, j_coord
            del i, i_coord
        linker_edge_list = [torch.LongTensor(sender_linker_nodes), torch.LongTensor(receiver_linker_nodes)] 

        print("getting fragment edge list")
        #### for the fragment part build the edge list considering that the fragment is FC
        sender_fragment_nodes, receiver_fragment_nodes = [], []
        for i, i_coord in enumerate(positions_fragment):
            for j, j_coord in enumerate(positions_fragment):
                sender_fragment_nodes.append(i)
                receiver_fragment_nodes.append(j)
                del j, j_coord
            del i, i_coord
        fragment_edge_list = [torch.LongTensor(sender_fragment_nodes), torch.LongTensor(receiver_fragment_nodes)] 
        
        print("add info to dict")
        #### add all this tensor in the out dictionary
        out["position_linker_gen"] = positions_linker
        out["category_linker_gen"] = category_linker
        #=======================
        out["position_fragment"] = positions_fragment
        out["category_fragment"] = category_fragment
        #=======================
        out["position_protein"] = positions_protein
        out["category_protein"] = category_protein
        #=======================
        out["num_linker_gen_nodes"] = num_linker_nodes
        out["num_fragment_nodes"] = num_fragment_nodes
        out["num_protein_nodes"] = num_protein_nodes
        #=======================
        out["linker_edge_list"] = linker_edge_list
        out["fragment_edge_list"] = fragment_edge_list
        out["protein_chopped_edge_list"] = protein_chopped_edge_list
        
        #### append the out dictionary to the list on dictionaries and delete all variables to free mem
        out_list.append(out)
        del instance, out, coordinates, one_hot, linker_flag, context_flag, category, positions_linker, category_linker, positions_context, category_context, mask_category_0_context, positions_protein
        del category_protein, positions_fragment, category_fragment, num_linker_nodes, num_fragment_nodes, num_protein_nodes , sender_protein_nodes, receiver_protein_nodes, protein_chopped_edge_list
        del sender_linker_nodes, receiver_linker_nodes, linker_edge_list, sender_fragment_nodes, receiver_fragment_nodes, fragment_edge_list
    del file, file_name #free mem
torch.save(out_list, "/home/piazza/markov_bridges/LP_Data/RestyledReducedDiffusionDataset/train_4A.pt") #save train
del out_list , id_counter , filename_list #free mem

print("Validation conversion")
out_list , id_counter = [] , 0 #empty list where to append all validation set molecule's dictionaries, counter for the number of training molecules
filename_list = os.listdir(val_in_path)
filename_list.sort() #sort pkl files
for file_name in filename_list: #for each val .pkl file
    print("processing pkl")
    file = dill.load(open(f"{val_in_path}/{file_name}", "rb"))
    for instance in file: #for each graph stored in the pkl file
        out = {} #create an empty dictionary which represent this instance; this dictionary will be appended to the out_list
        out["uuid"] = id_counter #instance number number
        id_counter += 1 #increment number
        out["name"] = instance["instanceFilename"] #track filename (contains also the rcsb id as prefix)
        
        #### retrieve coordinates, onehot encoding and linker/context flag
        coordinates = instance["x"][:,:3] #nodes coordinates are the first 3 columns. THOSE ARE ALL NODES COORDINATES
        one_hot = instance["x"][:, 3:-1] #one hot encoding of atom type. THIS IS THE ONEHOT FOR ALL NODES
        linker_flag = instance["x"][:, -1] #binary flag which discriminate linker nodes (1) from context nodex (0). THIS IS THE LINKER FLAG FOR ALL NODES
        context_flag = 1-linker_flag #binary flag which discriminate linker nodes (0) from context nodex (1). THIS IS THE CONTEXT FLAG FOR ALL NODES

        #### covert the onehot encoding into a single number (es [0,0,0,1,0] -> 3)
        category = torch.argmax(one_hot, dim=1) #Atom type retrieved from one hot encoding. atom type is 0 for exclusion spheres. THIS IS THE CATEGORY FOR ALL NODES
        
        #### get positions and category for linker atoms by discriminating linker atoms from context atoms using the linker flag
        positions_linker = coordinates[linker_flag.bool()] #use the binary flag to identify linker nodes, then retrieve only their positions
        category_linker = category[linker_flag.bool()] #use the binary flag to identify linker nodes, then retrieve only their category
        
        #### get positions and category for context atoms (context = fragment & protein) by using the context flag
        positions_context = coordinates[context_flag.bool()] #use binary flag to identify context nodes, ten retrieve only their position
        category_context = category[context_flag.bool()] #use binary flag to identify context nodes, then retireve only their category

        #### within the context, discriminate fragment from protein nodes using the category (0 is for protein, all other categories are fragment atoms)
        mask_category_0_context = (category_context == 0) #boolean mask that identify which context nodes belong to category 0 and which not
        positions_protein = positions_context[mask_category_0_context] #get position of only protein nodes
        category_protein = category_context[mask_category_0_context] #get the category for protein nodes (this wille full of 0)
        positions_fragment = positions_context[~mask_category_0_context] #get position of fragment nodes by the negation of the protein mask nodes
        category_fragment = category_context[~mask_category_0_context] #get categories of fragment nodes
        
        #### retrieve the number of linker nodes, fragment nodes and protein nodes
        num_linker_nodes, num_fragment_nodes, num_protein_nodes = category_linker.shape[0], category_fragment.shape[0], category_protein.shape[0]

        #### for the protein part, build the edge list by adding an edge only if the node distance is <= 2.5 A. the edge list for 1 protein has shape [2, num_existing_edges*2], the *2 is because the graph is undirected
        sender_protein_nodes, receiver_protein_nodes = [], [] #empty list where to store indexes of sender and receiver node of an edge in the protein part
        for i, i_coord in enumerate(positions_protein): #for each protein node and its coordinates
            for j, j_coord in enumerate(positions_protein): #with all other protein nodes (included itself) and their coordinates 
                dist = math.dist(i_coord, j_coord) #calculate euclidean distance between nodes
                if dist <= 4: #if the distance is below the threshold then the edge exist (in this way also self loop exists)
                    sender_protein_nodes.append(i) #i is sender (in this double iteration i will later be the receiver and  j the sender)
                    receiver_protein_nodes.append(j) #j is receiver
                del j, j_coord, dist #free mem in loop
            del i, i_coord #free mem in loop
        protein_chopped_edge_list = [torch.LongTensor(sender_protein_nodes), torch.LongTensor(receiver_protein_nodes)] #edge list after chopping edges above the threshold

        #### for the linker part, build the edge list considering that the linker is FC
        sender_linker_nodes, receiver_linker_nodes = [], []
        for i, i_coord in enumerate(positions_linker):
            for j, j_coord in enumerate(positions_linker):
                sender_linker_nodes.append(i)
                receiver_linker_nodes.append(j)
                del j, j_coord
            del i, i_coord
        linker_edge_list = [torch.LongTensor(sender_linker_nodes), torch.LongTensor(receiver_linker_nodes)] 

        #### for the fragment part build the edge list considering that the fragment is FC
        sender_fragment_nodes, receiver_fragment_nodes = [], []
        for i, i_coord in enumerate(positions_fragment):
            for j, j_coord in enumerate(positions_fragment):
                sender_fragment_nodes.append(i)
                receiver_fragment_nodes.append(j)
                del j, j_coord
            del i, i_coord
        fragment_edge_list = [torch.LongTensor(sender_fragment_nodes), torch.LongTensor(receiver_fragment_nodes)] 
        
        #### add all this tensor in the out dictionary
        out["position_linker_gen"] = positions_linker
        out["category_linker_gen"] = category_linker
        #=======================
        out["position_fragment"] = positions_fragment
        out["category_fragment"] = category_fragment
        #=======================
        out["position_protein"] = positions_protein
        out["category_protein"] = category_protein
        #=======================
        out["num_linker_gen_nodes"] = num_linker_nodes
        out["num_fragment_nodes"] = num_fragment_nodes
        out["num_protein_nodes"] = num_protein_nodes
        #=======================
        out["linker_edge_list"] = linker_edge_list
        out["fragment_edge_list"] = fragment_edge_list
        out["protein_chopped_edge_list"] = protein_chopped_edge_list
        
        #### append the out dictionary to the list on dictionaries and delete all variables to free mem
        out_list.append(out)
        del instance, out, coordinates, one_hot, linker_flag, context_flag, category, positions_linker, category_linker, positions_context, category_context, mask_category_0_context, positions_protein
        del category_protein, positions_fragment, category_fragment, num_linker_nodes, num_fragment_nodes, num_protein_nodes , sender_protein_nodes, receiver_protein_nodes, protein_chopped_edge_list
        del sender_linker_nodes, receiver_linker_nodes, linker_edge_list, sender_fragment_nodes, receiver_fragment_nodes, fragment_edge_list
    del file, file_name #free mem
torch.save(out_list, "/home/piazza/markov_bridges/LP_Data/RestyledReducedDiffusionDataset/validation_4A.pt") #save train
del out_list , id_counter , filename_list #free mem


print("Test conversion")
out_list , id_counter = [] , 0 #empty list where to append all validation set molecule's dictionaries, counter for the number of training molecules
filename_list = os.listdir(test_in_path)
filename_list.sort() #sort pkl files
for file_name in filename_list: #for each val .pkl file
    print("processing pkl")
    file = dill.load(open(f"{test_in_path}/{file_name}", "rb"))
    for instance in file: #for each graph stored in the pkl file
        out = {} #create an empty dictionary which represent this instance; this dictionary will be appended to the out_list
        out["uuid"] = id_counter #instance number number
        id_counter += 1 #increment number
        out["name"] = instance["instanceFilename"] #track filename (contains also the rcsb id as prefix)
        
        #### retrieve coordinates, onehot encoding and linker/context flag
        coordinates = instance["x"][:,:3] #nodes coordinates are the first 3 columns. THOSE ARE ALL NODES COORDINATES
        one_hot = instance["x"][:, 3:-1] #one hot encoding of atom type. THIS IS THE ONEHOT FOR ALL NODES
        linker_flag = instance["x"][:, -1] #binary flag which discriminate linker nodes (1) from context nodex (0). THIS IS THE LINKER FLAG FOR ALL NODES
        context_flag = 1-linker_flag #binary flag which discriminate linker nodes (0) from context nodex (1). THIS IS THE CONTEXT FLAG FOR ALL NODES

        #### covert the onehot encoding into a single number (es [0,0,0,1,0] -> 3)
        category = torch.argmax(one_hot, dim=1) #Atom type retrieved from one hot encoding. atom type is 0 for exclusion spheres. THIS IS THE CATEGORY FOR ALL NODES
        
        #### get positions and category for linker atoms by discriminating linker atoms from context atoms using the linker flag
        positions_linker = coordinates[linker_flag.bool()] #use the binary flag to identify linker nodes, then retrieve only their positions
        category_linker = category[linker_flag.bool()] #use the binary flag to identify linker nodes, then retrieve only their category
        
        #### get positions and category for context atoms (context = fragment & protein) by using the context flag
        positions_context = coordinates[context_flag.bool()] #use binary flag to identify context nodes, ten retrieve only their position
        category_context = category[context_flag.bool()] #use binary flag to identify context nodes, then retireve only their category

        #### within the context, discriminate fragment from protein nodes using the category (0 is for protein, all other categories are fragment atoms)
        mask_category_0_context = (category_context == 0) #boolean mask that identify which context nodes belong to category 0 and which not
        positions_protein = positions_context[mask_category_0_context] #get position of only protein nodes
        category_protein = category_context[mask_category_0_context] #get the category for protein nodes (this wille full of 0)
        positions_fragment = positions_context[~mask_category_0_context] #get position of fragment nodes by the negation of the protein mask nodes
        category_fragment = category_context[~mask_category_0_context] #get categories of fragment nodes
        
        #### retrieve the number of linker nodes, fragment nodes and protein nodes
        num_linker_nodes, num_fragment_nodes, num_protein_nodes = category_linker.shape[0], category_fragment.shape[0], category_protein.shape[0]

         #### for the protein part, build the edge list by adding an edge only if the node distance is <= 2.5 A. the edge list for 1 protein has shape [2, num_existing_edges*2], the *2 is because the graph is undirected
        sender_protein_nodes, receiver_protein_nodes = [], [] #empty list where to store indexes of sender and receiver node of an edge in the protein part
        for i, i_coord in enumerate(positions_protein): #for each protein node and its coordinates
            for j, j_coord in enumerate(positions_protein): #with all other protein nodes (included itself) and their coordinates 
                dist = math.dist(i_coord, j_coord) #calculate euclidean distance between nodes
                if dist <= 4: #if the distance is below the threshold then the edge exist (in this way also self loop exists)
                    sender_protein_nodes.append(i) #i is sender (in this double iteration i will later be the receiver and  j the sender)
                    receiver_protein_nodes.append(j) #j is receiver
                del j, j_coord, dist #free mem in loop
            del i, i_coord #free mem in loop
        protein_chopped_edge_list = [torch.LongTensor(sender_protein_nodes), torch.LongTensor(receiver_protein_nodes)] #edge list after chopping edges above the threshold

        #### for the linker part, build the edge list considering that the linker is FC
        sender_linker_nodes, receiver_linker_nodes = [], []
        for i, i_coord in enumerate(positions_linker):
            for j, j_coord in enumerate(positions_linker):
                sender_linker_nodes.append(i)
                receiver_linker_nodes.append(j)
                del j, j_coord
            del i, i_coord
        linker_edge_list = [torch.LongTensor(sender_linker_nodes), torch.LongTensor(receiver_linker_nodes)] 

        #### for the fragment part build the edge list considering that the fragment is FC
        sender_fragment_nodes, receiver_fragment_nodes = [], []
        for i, i_coord in enumerate(positions_fragment):
            for j, j_coord in enumerate(positions_fragment):
                sender_fragment_nodes.append(i)
                receiver_fragment_nodes.append(j)
                del j, j_coord
            del i, i_coord
        fragment_edge_list = [torch.LongTensor(sender_fragment_nodes), torch.LongTensor(receiver_fragment_nodes)] 
        
        #### add all this tensor in the out dictionary
        out["position_linker_gen"] = positions_linker
        out["category_linker_gen"] = category_linker
        #=======================
        out["position_fragment"] = positions_fragment
        out["category_fragment"] = category_fragment
        #=======================
        out["position_protein"] = positions_protein
        out["category_protein"] = category_protein
        #=======================
        out["num_linker_gen_nodes"] = num_linker_nodes
        out["num_fragment_nodes"] = num_fragment_nodes
        out["num_protein_nodes"] = num_protein_nodes
        #=======================
        out["linker_edge_list"] = linker_edge_list
        out["fragment_edge_list"] = fragment_edge_list
        out["protein_chopped_edge_list"] = protein_chopped_edge_list
        
        #### append the out dictionary to the list on dictionaries and delete all variables to free mem
        out_list.append(out)
        del instance, out, coordinates, one_hot, linker_flag, context_flag, category, positions_linker, category_linker, positions_context, category_context, mask_category_0_context, positions_protein
        del category_protein, positions_fragment, category_fragment, num_linker_nodes, num_fragment_nodes, num_protein_nodes , sender_protein_nodes, receiver_protein_nodes, protein_chopped_edge_list
        del sender_linker_nodes, receiver_linker_nodes, linker_edge_list, sender_fragment_nodes, receiver_fragment_nodes, fragment_edge_list
    del file, file_name #free mem
torch.save(out_list, "/home/piazza/markov_bridges/LP_Data/RestyledReducedDiffusionDataset/test_4A.pt") #save train
del out_list , id_counter , filename_list #free mem

print("Conversion finished.")