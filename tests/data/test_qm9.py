from dataclasses import dataclass

from markov_bridges.data.qm9.qm9_graph_dataloader import QM9Dataset,RemoveYTransform
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

from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config
from markov_bridges.data.dataloaders_utils import get_dataloaders
from markov_bridges.configs.experiments_configs.mixed.edmg_experiments import get_edmg_experiment

def test_digress_qm9():
    train = QM9Dataset(stage='train', 
                    root=r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\data\raw\graph", 
                    remove_h=False,
                    target_prop=None, 
                    transform=RemoveYTransform(),
                    max_num_mol=100)
    return train

def test_max_qm9():
    config = get_edmg_experiment()
    dataloaders = get_dataloaders(config)

    #databatch = dataloaders.get_databatch()
    print(dataloaders.get_databach_keys())

if __name__=="__main__":
    test_max_qm9()