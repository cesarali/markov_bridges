from dataclasses import dataclass


@dataclass
class BasicPipelineConfig:
    name:str="BasicPipeline"
    number_of_steps:int = 20
    num_intermediates:int = 10
    max_rate_at_end:bool = False
    time_epsilon:float = 0.05
