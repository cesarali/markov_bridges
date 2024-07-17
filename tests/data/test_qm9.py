from dataclasses import dataclass

from markov_bridges.data.qm9_dataloader import QM9DataModule
from markov_bridges.data.qm9_dataloader import QM9Dataset,RemoveYTransform
from markov_bridges.data.qm9.dataset import retrieve_dataloaders

"""
# Experiment settings
exp_name: debug_10
dataset: 'qm9'            # qm9, qm9_positional
filter_n_atoms: null      # When set to an integer value, QM9 will only contain molecules of that amount of atoms
n_report_steps: 1
wandb_usr: cvignac
no_cuda: False
wandb: False             # Use wandb?
online: True             # True: online / False: offline
data_dir: 'data'
"""


@dataclass
class QM9Config:
    batch_size:int = 32
    num_workers:int = 12
    filter_n_atoms:int = None
    include_charges:bool = True
    subtract_thermo:bool = False
    force_download:bool = False

    remove_h:bool = False
    dataset:str = 'qm9'
    datadir:str = r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\data\raw\graph"
    wandb:bool = False

def test_digress_qm9():
    train = QM9Dataset(stage='train', 
                    root=r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\data\raw\graph", 
                    remove_h=False,
                    target_prop=None, 
                    transform=RemoveYTransform(),
                    max_num_mol=100)
    return train

def test_max_qm9():
    config = QM9Config()
    
    dataloaders, charge_scale = retrieve_dataloaders(config)
    for databatch in dataloaders["train"]:
        print(databatch)
        break

if __name__=="__main__":
    test_max_qm9()