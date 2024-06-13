import os
from typing import List,Union
from pprint import pprint
from dataclasses import dataclass
from dataclasses import field,asdict
from markov_bridges import data_path


# model config
from markov_bridges.configs.config_classes.networks.temporal_networks_config import (
    TemporalMLPConfig,
    TemporalDeepMLPConfig,
    TemporalGraphConvNetConfig,
    ConvNetAutoencoderConfig,
    DiffusersUnet2DConfig,
    TemporalScoreNetworkAConfig,
    SequenceTransformerConfig,
    SimpleTemporalGCNConfig
)

from markov_bridges.configs.config_classes.pipelines.cjb_thermostat_configs import (
    LogThermostatConfig,
    ConstantThermostatConfig,
    ExponentialThermostatConfig,
    InvertedExponentialThermostatConfig
)

from markov_bridges.configs.config_classes.trainers.trainer_config import BasicTrainerConfig

data_configs = {"NISTLoader":NISTLoaderConfig,
                "DistortedNISTLoader":DistortedNISTLoaderConfig,
                "LakhPianoRoll":LakhPianoRollConfig,
                "GraphDataloader":GraphDataloaderConfig,
                "BridgeConfig":BridgeConfig,
                "StatesDataloader":StatesDataloaderConfig,
                "GrayCodesDataloader":GrayCodesDataloaderConfig}


temporal_network_configs = {
    "TemporalMLP":TemporalMLPConfig,
    "ConvNetAutoencoder":ConvNetAutoencoderConfig,
    "SequenceTransformer":SequenceTransformerConfig,
    "TemporalDeepMLP":TemporalDeepMLPConfig,
    "TemporalGraphConvNet":TemporalGraphConvNetConfig,
    "TemporalScoreNetworkA":TemporalScoreNetworkAConfig,
    "DiffusersUnet2D":DiffusersUnet2DConfig,
    "SimpleTemporalGCN":SimpleTemporalGCNConfig
}

thermostat_configs = {
    "LogThermostat":LogThermostatConfig,
    "ConstantThermostat":ConstantThermostatConfig,
    "ExponentialThermostat":ExponentialThermostatConfig,
    "InvertedExponentialThermostat":InvertedExponentialThermostatConfig
}

image_data_path = os.path.join(data_path,"raw")

@dataclass
class CJBTrainerConfig(BasicTrainerConfig):
    name:str = "CRMTrainer"
    loss_regularize_variance:bool = False
    loss_regularize:bool = False
    loss_regularize_square:bool = False
    max_iterations:int = 1000000
    warm_up:int=0
    
@dataclass
class OptimalTransportSamplerConfig:
    name: str = "uniform" # uniform,OTPlanSampler
    method: str = "exact" 
    cost: str = None #log, None
    reg: float = 0.05
    reg_m: float = 1.0
    normalize_cost: bool = False
    normalize_cost_constant: float = 1.
    warn: bool = True

    def __post_init__(self):
        if self.name == "uniform":
            self.cost = None
            
        if self.cost == "log":
            self.method = "sinkhorn"
        
@dataclass
class BasicPipelineConfig:
    name:str="BasicPipeline"
    number_of_steps:int = 20
    num_intermediates:int = 10
    time_epsilon = 0.05
    set_diagonal = True

@dataclass
class TemporalNetworkToRateConfig:
    name:str = "TemporalNetworkToRate"
    type_of:str = None # bernoulli, empty, linear,logistic, None
    linear_reduction:Union[float,int] = 0.1 # if None full layer, 
                                            # if float is the percentage of output dimensions that is assigned as hidden 
                                            #otherwise hidden
    fix_logistic = True

@dataclass
class CJBConfig:
    # data
    data0: Union[LakhPianoRollConfig,StatesDataloaderConfig] = StatesDataloaderConfig()
    data1: Union[LakhPianoRollConfig,NISTLoaderConfig] = NISTLoaderConfig()
    # process
    thermostat : Union[ConstantThermostatConfig, LogThermostatConfig] = ConstantThermostatConfig()
    # temporal_to_rate
    temporal_network_to_rate : Union[int,float,TemporalNetworkToRateConfig] = None
    # temporal network
    temporal_network: Union[TemporalMLPConfig,
                            ConvNetAutoencoderConfig,
                            DiffusersUnet2DConfig,
                            TemporalScoreNetworkAConfig,
                            SequenceTransformerConfig,
                            SimpleTemporalGCNConfig] = TemporalMLPConfig()
    # ot
    optimal_transport:OptimalTransportSamplerConfig = OptimalTransportSamplerConfig()
    # training
    trainer: CJBTrainerConfig = CJBTrainerConfig()
    #pipeline
    pipeline : BasicPipelineConfig = BasicPipelineConfig()

    def __post_init__(self):
        if isinstance(self.data0,dict):
            self.data0 = data_configs[self.data0["name"]](**self.data0)

        if isinstance(self.data1,dict):
            self.data1 = data_configs[self.data1["name"]](**self.data1)

        if isinstance(self.temporal_network,dict):
            self.temporal_network = temporal_network_configs[self.temporal_network["name"]](**self.temporal_network)

        if isinstance(self.optimal_transport,dict):
            self.optimal_transport = OptimalTransportSamplerConfig(**self.optimal_transport)

        if isinstance(self.thermostat, dict):
            self.thermostat = thermostat_configs[self.thermostat["name"]](**self.thermostat)

        if isinstance(self.trainer,dict):
            self.trainer = CJBTrainerConfig(**self.trainer)

        if isinstance(self.pipeline,dict):
            self.pipeline = BasicPipelineConfig(**self.pipeline)

        if isinstance(self.temporal_network_to_rate,dict):
            self.temporal_network_to_rate = TemporalNetworkToRateConfig(**self.temporal_network_to_rate)
