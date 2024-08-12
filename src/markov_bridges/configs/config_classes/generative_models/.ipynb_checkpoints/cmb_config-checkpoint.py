import os
from typing import List,Union
from pprint import pprint
from dataclasses import dataclass, field
from dataclasses import field,asdict
from markov_bridges import data_path


# model config
from markov_bridges.configs.config_classes.networks.mixed_networks_config import (
    MixedDeepMLPConfig
)

from markov_bridges.configs.config_classes.pipelines.cjb_thermostat_configs import (
    LogThermostatConfig,
    ConstantThermostatConfig,
    ExponentialThermostatConfig,
    InvertedExponentialThermostatConfig
)

from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.data.graphs_configs import GraphDataloaderGeometricConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig, GaussiansConfig
from markov_bridges.configs.config_classes.pipelines.pipeline_configs import CMBPipelineConfig
from markov_bridges.configs.config_classes.trainers.trainer_config import CMBTrainerConfig

image_data_path = os.path.join(data_path,"raw")

data_configs = {"LakhPianoRoll":LakhPianoRollConfig,
                "GraphDataloaderGeometric":GraphDataloaderGeometricConfig,
                "IndependentMix":IndependentMixConfig,
                "Gaussians":GaussiansConfig}

mixed_network_configs = {
    "MixedDeepMLP":MixedDeepMLPConfig,
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
class CMBConfig:
    """
    Data class to store all configuration files from CMB model
    """
    # process config
    continuous_loss_type:str = "flow" # flow, regression, drift
    brownian_sigma: float = 0.1
    # data
    data: LakhPianoRollConfig|IndependentMixConfig = field(default_factory=IndependentMixConfig)#lp
    # process 
    thermostat : Union[ConstantThermostatConfig, LogThermostatConfig] = field(default_factory=ConstantThermostatConfig)#lp
    # temporal network
    mixed_network: Union[MixedDeepMLPConfig] = field(default_factory=MixedDeepMLPConfig)#lp
    # ot
    optimal_transport:OptimalTransportSamplerConfig = field(default_factory=OptimalTransportSamplerConfig)#lp
    # training
    trainer: CMBTrainerConfig = field(default_factory=CMBTrainerConfig)#lp
    #pipeline
    pipeline : CMBPipelineConfig = field(default_factory=CMBPipelineConfig(num_intermediates=None))#lp

    def __post_init__(self):
        if isinstance(self.data,dict):
            self.data = data_configs[self.data["name"]](**self.data)

        if isinstance(self.mixed_network,dict):
            self.mixed_network = mixed_network_configs[self.mixed_network["name"]](**self.mixed_network)

        if isinstance(self.optimal_transport,dict):
            self.optimal_transport = OptimalTransportSamplerConfig(**self.optimal_transport)

        if isinstance(self.thermostat, dict):
            self.thermostat = thermostat_configs[self.thermostat["name"]](**self.thermostat)

        if isinstance(self.trainer,dict):
            self.trainer = CMBTrainerConfig(**self.trainer)

        if isinstance(self.pipeline,dict):
            self.pipeline = CMBPipelineConfig(**self.pipeline)

