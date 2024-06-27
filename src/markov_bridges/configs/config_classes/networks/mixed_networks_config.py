from dataclasses import dataclass

@dataclass
class MixedDeepMLPConfig:
    name : str = "MixedDeepMLP"
    time_embed_dim : int = 50
    discrete_embed_dim: int = 50
    hidden_dim : int = 250
    activation : str = 'ReLU'
    num_layers : int = 4
    ema_decay: float = 0.999
    dropout : float = 0.2