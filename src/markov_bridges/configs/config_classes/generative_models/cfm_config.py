import os
from typing import List,Union
from pprint import pprint
from dataclasses import dataclass
from dataclasses import field,asdict
from markov_bridges import data_path


# model config
from markov_bridges.configs.config_classes.networks.continuous_network_config import DeepMLPConfig
from markov_bridges.configs.config_classes.pipelines.cjb_thermostat_configs import ConstantThermostatConfig


from markov_bridges.configs.config_classes.data.basics_configs import GaussiansConfig
from markov_bridges.configs.config_classes.pipelines.pipeline_configs import CFMPipelineConfig
from markov_bridges.configs.config_classes.trainers.trainer_config import CFMTrainerConfig

image_data_path = os.path.join(data_path,"raw")

data_configs = {"Gaussians": GaussiansConfig}
continuous_network_configs = {"DeepMLP": DeepMLPConfig}
thermostat_configs = {"ConstantThermostat": ConstantThermostatConfig}

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
class CFMConfig:
    """
    Data class to store all configuration files from CMB model
    """
    # data
    data: GaussiansConfig = GaussiansConfig()
    # process 
    thermostat: ConstantThermostatConfig = ConstantThermostatConfig()
    # temporal network
    continuous_network: DeepMLPConfig = DeepMLPConfig()
    # ot
    optimal_transport: OptimalTransportSamplerConfig = OptimalTransportSamplerConfig()
    # training
    trainer: CFMTrainerConfig = CFMTrainerConfig()
    #pipeline
    pipeline: CFMPipelineConfig = CFMPipelineConfig(num_intermediates=None)

    def __post_init__(self):
        if isinstance(self.data, dict):
            self.data = data_configs[self.data["name"]](**self.data)

        if isinstance(self.continuous_network, dict):
            self.continuous_network = continuous_network_configs[self.continuous_network["name"]](**self.continuous_network)

        if isinstance(self.optimal_transport, dict):
            self.optimal_transport = OptimalTransportSamplerConfig(**self.optimal_transport)

        if isinstance(self.thermostat, dict):
            self.thermostat = thermostat_configs[self.thermostat["name"]](**self.thermostat)

        if isinstance(self.trainer, dict):
            self.trainer = CFMTrainerConfig(**self.trainer)

        if isinstance(self.pipeline, dict):
            self.pipeline = CFMPipelineConfig(**self.pipeline)

