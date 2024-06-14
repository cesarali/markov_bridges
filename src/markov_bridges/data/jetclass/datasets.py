import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass

from DynGenModels.datamodules.jetclass.dataprocess import PreProcessJetClassData

class JetClassDataset(Dataset):

    def __init__(self, config):
        self.data_source = config.DATA_SOURCE
        self.data_target = config.DATA_TARGET
        self.dim_input = config.DIM_INPUT
        self.features = config.FEATURES
        self.max_num_constituents = config.MAX_NUM_CONSTITUENTS
        self.preprocess_methods = config.PREPROCESS
        self.get_target_data()
        self.get_source_data()

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.target_preprocess[idx]
        output['source'] = self.source_preprocess[idx]
        output['target context'] = self.jets_target[idx]
        output['source context'] = self.jets_source[idx]
        output['context'] = torch.zeros_like(output['target'][..., None:1]) #...to be replaced with jet label
        output['traget mask'] = torch.ones_like(output['target'][..., None:1])
        output['source mask'] = torch.ones_like(output['source'][..., None:1])
        return output

    def __len__(self):
        return len(self.target)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_target_data(self):
        if self.data_target == 'qcd': f = h5py.File('/home/df630/DynGenModels/data/jetclass/qcd_top_jets/qcd_N30_100k.hdf5', 'r') 
        elif self.data_target == 'top': f = h5py.File('/home/df630/DynGenModels/data/jetclass/qcd_top_jets/top_N30_100k.hdf5', 'r')
        self.constituents = torch.Tensor(np.array(f['4_momenta']))[..., :4]
        self.jets = torch.sum(self.constituents, dim=1)
        self.jets_target = self.get_features(self.jets)
        jet_axis = self.jets.unsqueeze(1).repeat(1, self.max_num_constituents, 1)        
        data = PreProcessJetClassData(self.get_features(self.constituents, axis=jet_axis, flatten_tensor=True), 
                                      contituents=True, 
                                      methods=self.preprocess_methods)      

        self.target = data.features.clone()
        data.preprocess()
        self.summary_stats = data.summary_stats
        self.target_preprocess = data.features.clone()

    def get_source_data(self):
        if self.data_source == 'noise': 
            self.source = torch.randn_like(self.constituents)
            self.jets_source = torch.randn_like(self.jets)
            data = PreProcessJetClassData(self.source, 
                                          contituents=True, 
                                          methods=self.preprocess_methods)  
            data.preprocess()
            self.source_preprocess = data.features.clone()
        else:
            if self.data_source == 'qcd': f = h5py.File('/home/df630/DynGenModels/data/jetclass/qcd_top_jets/qcd_N30_100k.hdf5', 'r') 
            elif self.data_source == 'top': f = h5py.File('/home/df630/DynGenModels/data/jetclass/qcd_top_jets/top_N30_100k.hdf5', 'r')
            constituents = torch.Tensor(np.array(f['4_momenta']))[..., :4]
            jets = torch.sum(constituents, dim=1)
            self.jets_source = self.get_features(jets)
            jet_axis = jets.unsqueeze(1).repeat(1, self.max_num_constituents, 1)        
            data = PreProcessJetClassData(self.get_features(constituents, axis=jet_axis, flatten_tensor=True), 
                                        contituents=True, 
                                        methods=self.preprocess_methods)  
            self.source = data.features.clone()
            data.preprocess()
            self.source_preprocess = data.features.clone()


        
    def get_features(self, four_mom, axis=None, flatten_tensor=False):
        four_mom = four_mom.reshape(-1,4) if flatten_tensor else four_mom
        px, py, pz, e = four_mom[...,0], four_mom[...,1], four_mom[...,2], four_mom[...,3]
        pt = torch.sqrt(px**2 + py**2)
        eta = 0.5 * np.log( (e + pz) / (e - pz))
        phi = np.arctan2(py, px)
        m = torch.sqrt(e**2 - px**2 - py**2 - pz**2)
        
        if axis is not None:
            axis = axis.reshape(-1,4) if flatten_tensor else axis
            px_axis, py_axis, pz_axis, e_axis = axis[...,0], axis[...,1], axis[...,2], axis[...,3]
            pt_axis = torch.sqrt(px_axis**2 + py_axis**2)
            eta_axis = 0.5 * np.log( (e_axis + pz_axis) / (e_axis - pz_axis))
            phi_axis= np.arctan2(py_axis, px_axis)
            m_axis = torch.sqrt(e_axis**2 - px_axis**2 - py_axis**2 - pz_axis**2)
            #...coords relative to axis:
            pt = pt / pt_axis
            eta = eta - eta_axis
            phi = (phi - phi_axis + np.pi) % (2 * np.pi) - np.pi
            m = m / m_axis

        hadr_coord = torch.stack([pt, eta, phi, m], dim=1)
        output = hadr_coord.reshape(-1, self.max_num_constituents, 4) if flatten_tensor else hadr_coord

        return output
