
from torch.utils.data import DataLoader
from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config
from markov_bridges.data.qm9.data.args import init_argparse
from markov_bridges.data.qm9.data.collate import PreprocessQM9
from markov_bridges.data.qm9.data.utils import initialize_datasets
import os

def retrieve_dataloaders(cfg:QM9Config):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    filter_n_atoms = cfg.filter_n_atoms
    # Initialize dataloader
    args = init_argparse('qm9')
    # data_dir = cfg.data_root_dir
    keys,datasets, num_species, charge_scale = initialize_datasets(cfg.datadir, 
                                                                    cfg.dataset,
                                                                    subtract_thermo=cfg.subtract_thermo,
                                                                    force_download=cfg.force_download,
                                                                    remove_h=cfg.remove_h,
                                                                    num_pts_train= cfg.num_pts_train,
                                                                    num_pts_valid=cfg.num_pts_valid,
                                                                    num_pts_test=cfg.num_pts_test)
    
    qm9_to_eV = {'U0': 27.2114, 
                 'U': 27.2114, 
                 'G': 27.2114, 
                 'H': 27.2114, 
                 'zpve': 27211.4, 
                 'gap': 27.2114, 
                 'homo': 27.2114,
                 'lumo': 27.2114}

    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)

    if filter_n_atoms is not None:
        print("Retrieving molecules with only %d atoms" % filter_n_atoms)
        datasets = filter_atoms(datasets, filter_n_atoms)

    # Construct PyTorch dataloaders from datasets
    preprocess = PreprocessQM9(load_charges=cfg.include_charges)
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=preprocess.collate_fn)
                            for split, dataset in datasets.items()}
        
    return keys,dataloaders, charge_scale

def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets