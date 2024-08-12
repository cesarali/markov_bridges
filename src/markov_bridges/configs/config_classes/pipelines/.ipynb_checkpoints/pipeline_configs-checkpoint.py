from dataclasses import dataclass


@dataclass
class BasicPipelineConfig:
    name:str="BasicPipeline"
    number_of_steps:int = 20
    num_intermediates:int = 10
    max_rate_at_end:bool = False
    time_epsilon:float = 0.05


@dataclass
class CFMPipelineConfig(BasicPipelineConfig):
    ode_solver: str = 'euler'
    sensitivity: bool = 'autograd'
    atol: float = 1e-4
    rtol: float = 1e-4

@dataclass
class CMBPipelineConfig(BasicPipelineConfig):
    solver: str = 'ode_tau' #sde_tau
    sensitivity: bool = 'autograd'
    atol: float = 1e-4
    rtol: float = 1e-4