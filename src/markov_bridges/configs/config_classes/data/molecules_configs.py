from dataclasses import dataclass

@dataclass
class QM9Config:
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
    datadir:str = r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\data\raw\graph"
    wandb:bool = False