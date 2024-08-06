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
    SequenceTransformerConfig,
    SimpleTemporalGCNConfig
)

from markov_bridges.configs.config_classes.pipelines.cjb_thermostat_configs import (
    LogThermostatConfig,
    ConstantThermostatConfig,
    ExponentialThermostatConfig,
    InvertedExponentialThermostatConfig
)

from markov_bridges.configs.config_classes.pipelines.pipeline_configs import BasicPipelineConfig
from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.data.sequences_config import SinusoidalConfig
from markov_bridges.configs.config_classes.data.graphs_configs import GraphDataloaderGeometricConfig

from markov_bridges.configs.config_classes.trainers.trainer_config import CJBTrainerConfig
image_data_path = os.path.join(data_path,"raw")

data_configs = {"LakhPianoRoll":LakhPianoRollConfig,
                "Sinusoidal":SinusoidalConfig,
                "GraphDataloaderGeometric":GraphDataloaderGeometricConfig}

temporal_network_configs = {
    "TemporalMLP":TemporalMLPConfig,
    "SequenceTransformer":SequenceTransformerConfig,
    "TemporalDeepMLP":TemporalDeepMLPConfig,
    "SimpleTemporalGCN":SimpleTemporalGCNConfig
}

thermostat_configs = {
    "LogThermostat":LogThermostatConfig,
    "ConstantThermostat":ConstantThermostatConfig,
    "ExponentialThermostat":ExponentialThermostatConfig,
    "InvertedExponentialThermostat":InvertedExponentialThermostatConfig
}

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
class TemporalNetworkToRateConfig:
    name:str = "TemporalNetworkToRate"
    type_of:str = None # bernoulli, empty, linear,logistic, None
    linear_reduction:Union[float,int] = 0.1 # if None full layer, 
                                            # if float is the percentage of output dimensions that is assigned as hidden 
                                            #otherwise hidden
    fix_logistic = True

@dataclass
class CJBConfig:
    """
    Data class to store all configuration files from CJB model
    """
    # data
    data: Union[LakhPianoRollConfig] = None
    # process 
    thermostat : Union[ConstantThermostatConfig, LogThermostatConfig] = ConstantThermostatConfig()
    # temporal_to_rate
    temporal_network_to_rate : Union[int,float,TemporalNetworkToRateConfig] = None
    # temporal network
    temporal_network: Union[TemporalMLPConfig,
                            SequenceTransformerConfig,
                            SimpleTemporalGCNConfig] = TemporalMLPConfig()
    # ot
    optimal_transport:OptimalTransportSamplerConfig = OptimalTransportSamplerConfig()
    # training
    trainer: CJBTrainerConfig = CJBTrainerConfig()
    #pipeline
    pipeline : BasicPipelineConfig = BasicPipelineConfig()

    def __post_init__(self):
        """
        In order to read dataclasses which are stored as dict down in the hierarchy, 
        one must instantiate from the dictionary.
        """
        if isinstance(self.data,dict):
            self.data = data_configs[self.data["name"]](**self.data)

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
