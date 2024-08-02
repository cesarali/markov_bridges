from dataclasses import dataclass


@dataclass
class DeepMLPConfig:
    name : str = "DeepMLP"
    time_embed_dim : int = 32
    discrete_embed_dim : int = 16
    hidden_dim : int = 128
    activation : str = 'ReLU'
    num_layers : int = 3
    ema_decay: float = 0.999
    dropout : float = 0.2