from typing import List
from dataclasses import dataclass,field

from markov_bridges import project_path
from markov_bridges.configs.config_classes.metrics.metrics_configs import metrics_config

@dataclass
class BasicTrainerConfig:
    epoch:int = 0
    number_of_training_steps:int = 0
    number_of_test_step:int = 0

    # Scheduler-specific parameters
    scheduler:str = None # step,reduce,exponential,multi

    step_size: int = 30  # for StepLR
    gamma: float = 0.1  # for StepLR, MultiStepLR, ExponentialLR
    milestones: List[int] = field(default_factory=lambda: [50, 100, 150])  # for MultiStepLR
    factor: float = 0.1  # for ReduceLROnPlateau
    patience: int = 10  # for ReduceLROnPlateau

    number_of_epochs:int = 300
    log_loss:int = 100
    warm_up_best_model_epoch:int = 0
    save_model_test_stopping:bool = True
    save_model_metrics_stopping:bool = False
    save_model_metrics_warming:int = 10
    metric_to_save:str=None

    save_model_epochs:int = 1e6
    save_metric_epochs:int = 1e6
    max_test_size:int = 2000
    do_ema:bool = True
    clip_grad:bool = False
    clip_max_norm:float = 1.

    learning_rate:float = 0.001
    weight_decay:float =  0.0001
    lr_decay:float =  0.999

    accelerator: str="gpu"
    devices: str="3"
    strategy: str="auto"

    distributed: bool=False

    windows: bool = True
    berlin: bool = True
    debug:bool = False


    metrics: List[str] = field(default_factory=lambda :["mse_histograms",
                                                        "kdmm",
                                                        "categorical_histograms"])
    def __post_init__(self):
        self.berlin = self.windows
        self.save_model_epochs = int(.5*self.number_of_epochs)
        self.save_metric_epochs = self.number_of_epochs - 1
        self.save_model_test_stopping = not self.save_model_metrics_stopping


@dataclass
class CJBTrainerConfig(BasicTrainerConfig):
    name:str = "CJBTrainer"
    warm_up:int=0

    def __post_init__(self):
        new_metrics = []
        for metric in self.metrics:
            if isinstance(metric,dict):
                metric = metrics_config[metric["name"]](**metric)
                new_metrics.append(metric)
            elif isinstance(metric,str):
                new_metrics.append(metric)
            else:
                pass
        self.metrics = new_metrics


@dataclass
class CMBTrainerConfig(BasicTrainerConfig):
    name:str = "CMBTrainer"
    max_iterations:int = 1000000
    warm_up:int=0

    def __post_init__(self):
        new_metrics = []
        for metric in self.metrics:
            if isinstance(metric,dict):
                metric = metrics_config[metric["name"]](**metric)
                new_metrics.append(metric)
            elif isinstance(metric,str):
                new_metrics.append(metric)
            else:
                pass
        self.metrics = new_metrics

@dataclass
class EDMGTrainerConfig(BasicTrainerConfig):
    name:str = "EDMGTrainer"
    weight_decay:float = 1e-12
    amsgrad:bool = True
    warm_up:int=0

    def __post_init__(self):
        new_metrics = []
        for metric in self.metrics:
            if isinstance(metric,dict):
                metric = metrics_config[metric["name"]](**metric)
                new_metrics.append(metric)
            elif isinstance(metric,str):
                new_metrics.append(metric)
            else:
                pass
        self.metrics = new_metrics



@dataclass
class CFMTrainerConfig(BasicTrainerConfig):
    name: str = "CFMTrainer"
    conditional_bridge_type: str = "linear" # "schrodinger"
    loss_regularize_variance: bool = False
    loss_regularize: bool = False
    loss_regularize_square: bool = False
    max_iterations: int = 1000000
    warm_up: int = 0
    paralellize_gpu: bool=False

    def __post_init__(self):
        new_metrics = []
        for metric in self.metrics:
            if isinstance(metric,dict):
                metric = metrics_config[metric["name"]](**metric)
                new_metrics.append(metric)
            elif isinstance(metric,str):
                new_metrics.append(metric)
            else:
                pass
        self.metrics = new_metrics
