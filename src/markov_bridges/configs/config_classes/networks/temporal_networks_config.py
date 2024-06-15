from typing import List,Tuple
from dataclasses import dataclass,asdict,field

@dataclass
class TemporalDeepMLPConfig:
    name : str = "TemporalDeepMLP"
    time_embed_dim : int = 50
    hidden_dim : int = 250
    activation : str = 'ReLU'
    num_layers : int = 4
    ema_decay: float = 0.999
    dropout : float = 0.2

@dataclass
class TemporalMLPConfig:
    name:str = "TemporalMLP"
    time_embed_dim :int = 100
    hidden_dim :int = 100
    ema_decay :float = 0.9999  # 0.9999

@dataclass
class TemporalUNetConfig:
    name : str = "TemporalUNet"
    time_embed_dim : int = 128
    hidden_dim : int = 256
    ema_decay: float = 0.999
    dropout : float = 0.1
    activation : str = 'GELU'

@dataclass
class SequenceTransformerConfig:
    name: str = "SequenceTransformer"
    d_model:int = 128
    num_layers:int = 6
    num_heads:int = 8
    dim_feedforward:int = 2048
    dropout:float = 0.1
    temb_dim:int = 128
    num_output_FFresiduals:int = 2
    time_scale_factor:int = 1000
    use_one_hot_input:bool = True

    ema_decay :float = 0.9999  # 0.9999

@dataclass
class SimpleTemporalGCNConfig:
    name:str = "SimpleTemporalGCN"
    time_embed_dim:int = 19
    hidden_channels:int = 64
    ema_decay :float = 0.9999  # 0.9999


