import os
from typing import List,Union
from pprint import pprint
from dataclasses import dataclass
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


from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from markov_bridges.configs.config_classes.pipelines.pipeline_configs import BasicPipelineConfig
from markov_bridges.configs.config_classes.trainers.trainer_config import CMBTrainerConfig

image_data_path = os.path.join(data_path,"raw")

data_configs = {"LakhPianoRoll":QM9Config,
                "IndependentMix":IndependentMixConfig}

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
class NoisingModelConfig:
    # Model settings
    model: str = 'egnn_dynamics'     # 'our_dynamics | schnet | simple_dynamics | kernel_dynamics | egnn_dynamics | gnn_dynamics'
    probabilistic_model: str = 'diffusion'         # 'flow | diffusion | diffusion_ddpm'
    diffusion_steps: int = 200
    diffusion_noise_schedule: str = 'learned'
    diffusion_loss_type: str = 'vlb'
    n_layers: int = 6
    nf: int = 64          # Layer size
    ode_regularization: float = 0.001
    trace: str = 'hutch'        # 'hutch | exact'
    dequantization: str = 'argmax_variational'        # 'uniform | variational | argmax_variational | deterministic'
    tanh: bool = True                                  # 'use tanh in the coord_mlp'
    attention: bool = True                             # 'use attention in the EGNN'
    x_aggregation: str = 'sum'                        # 'sum | mean'
    conditioning: List[str] = None    # 'multiple arguments can be passed including: homo | onehot | lumo | num_atoms | etc. '
                                     # usage: "conditioning=['H_thermo', 'homo', 'onehot', 'H_thermo']"
    actnorm: bool = True
    norm_constant: float = 1             # diff/(|diff| + norm_constant)
    
    def __post_init__(self):
        if self.conditioning is None:
            self.conditioning = []
    
@dataclass
class EDMGConfig:
    """
    Data class to store all configuration files from CMB model
    """
    # data
    data: QM9Config|IndependentMixConfig = IndependentMixConfig()
    # temporal network
    noising_model: Union[MixedDeepMLPConfig] = MixedDeepMLPConfig()
    # training
    trainer: CMBTrainerConfig = CMBTrainerConfig()
    #pipeline
    pipeline : BasicPipelineConfig = BasicPipelineConfig(num_intermediates=None)

    def __post_init__(self):
        if isinstance(self.data,dict):
            self.data = data_configs[self.data["name"]](**self.data)

        if isinstance(self.noising_model,dict):
            self.noising_model = mixed_network_configs[self.noising_model["name"]](**self.noising_model)

        if isinstance(self.trainer,dict):
            self.trainer = CMBTrainerConfig(**self.trainer)

        if isinstance(self.pipeline,dict):
            self.pipeline = BasicPipelineConfig(**self.pipeline)

