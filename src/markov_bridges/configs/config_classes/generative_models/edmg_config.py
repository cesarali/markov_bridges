import os
from typing import List,Union
from pprint import pprint
from dataclasses import dataclass
from dataclasses import field,asdict
from markov_bridges import data_path
from typing import List, Optional, Union

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
from markov_bridges.configs.config_classes.trainers.trainer_config import EDMGTrainerConfig

image_data_path = os.path.join(data_path,"raw")




@dataclass
class NoisingModelConfig:
    # egnn dynammics
    model: str = 'egnn_dynamics' # 'our_dynamics | schnet | simple_dynamics | kernel_dynamics | egnn_dynamics | gnn_dynamics'
    nf: int = 128
    n_layers: int = 6
    attention: bool = True
    tanh: bool = True
    inv_sublayers: int = 1
    norm_constant: float = 1
    sin_embedding: bool = False
    normalization_factor: float = 1
    aggregation_method: str = 'sum'

    ema_decay:float = 0.999
    # noising parameters
    condition_time:bool = True
    probabilistic_model: str = 'diffusion'         # 'flow | diffusion | diffusion_ddpm'
    diffusion_steps: int = 200
    diffusion_noise_schedule: str = 'learned'
    diffusion_loss_type: str = 'vlb'
    diffusion_noise_precision: float = 1e-5
    normalize_factors: List[Union[int, float]] = field(default_factory=lambda: [1, 4, 1])

    # loss and samples
    ode_regularization: float = 0.001
    trace: str = 'hutch'        # 'hutch | exact'
    dequantization: str = 'argmax_variational'        # 'uniform | variational | argmax_variational | deterministic'
    x_aggregation: str = 'sum'                        # 'sum | mean'
    conditioning: List[str] = None    # 'multiple arguments can be passed including: homo | onehot | lumo | num_atoms | etc. '
                                     # usage: "conditioning=['H_thermo', 'homo', 'onehot', 'H_thermo']"
    actnorm: bool = True
    norm_constant: float = 1             # diff/(|diff| + norm_constant)
    context_node_nf:int =  0

    augment_noise:int = 0
    data_augmentation:bool = False

    def __post_init__(self):
        if self.conditioning is None:
            self.conditioning = []

@dataclass
class Config:
    exp_name: str = 'debug_10'
    model: str = 'egnn_dynamics'
    probabilistic_model: str = 'diffusion'
    diffusion_steps: int = 500
    diffusion_noise_schedule: str = 'polynomial_2'
    diffusion_noise_precision: float = 1e-5
    diffusion_loss_type: str = 'l2'
    n_epochs: int = 200
    batch_size: int = 128
    lr: float = 2e-4
    brute_force: bool = False
    actnorm: bool = True
    break_train_epoch: bool = False
    dp: bool = True
    condition_time: bool = True
    clip_grad: bool = True
    trace: str = 'hutch'
    n_layers: int = 6
    inv_sublayers: int = 1  

    nf: int = 128
    tanh: bool = True
    attention: bool = True
    norm_constant: float = 1  
    sin_embedding: bool = False
    ode_regularization: float = 1e-3
    dataset: str = 'qm9'
    datadir: str = 'qm9/temp'
    filter_n_atoms: Optional[int] = None
    dequantization: str = 'argmax_variational'
    n_report_steps: int = 1
    wandb_usr: Optional[str] = None
    no_wandb: bool = False
    online: bool = True
    no_cuda: bool = False
    save_model: bool = True
    generate_epochs: int = 1
    num_workers: int = 0
    test_epochs: int = 10
    data_augmentation: bool = False
    conditioning: List[str] = field(default_factory=list)
    resume: Optional[str] = None
    start_epoch: int = 0
    ema_decay: float = 0.999
    augment_noise: float = 0
    n_stability_samples: int = 500
    normalize_factors: List[Union[int, float]] = field(default_factory=lambda: [1, 4, 1])
    remove_h: bool = False
    include_charges: bool = True
    visualize_every_batch: int = int(1e8)
    normalization_factor: float = 1
    aggregation_method: str = 'sum'

@dataclass
class EDMGConfig:
    """
    Data class to store all configuration files from CMB model
    """
    # data
    data: QM9Config|IndependentMixConfig = QM9Config()
    # noising_model
    noising_model: Union[NoisingModelConfig] = NoisingModelConfig()
    # training
    trainer: EDMGTrainerConfig = EDMGTrainerConfig()
    #pipeline
    pipeline : BasicPipelineConfig = BasicPipelineConfig(num_intermediates=None)

    def __post_init__(self):
        if isinstance(self.data,dict):
            self.data = QM9Config(**self.data)

        if isinstance(self.noising_model,dict):
            self.noising_model = NoisingModelConfig(**self.noising_model)

        if isinstance(self.trainer,dict):
            self.trainer = EDMGTrainerConfig(**self.trainer)

        if isinstance(self.pipeline,dict):
            self.pipeline = BasicPipelineConfig(**self.pipeline)

