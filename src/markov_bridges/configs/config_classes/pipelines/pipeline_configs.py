from dataclasses import dataclass


@dataclass
class BasicPipelineConfig:
    name:str="BasicPipeline"
    number_of_steps:int = 20
    num_intermediates:int = 10
    time_epsilon = 0.05
    set_diagonal = True