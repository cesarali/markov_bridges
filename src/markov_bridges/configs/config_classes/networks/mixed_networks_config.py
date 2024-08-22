from typing import List
from dataclasses import dataclass

@dataclass
class MixedDeepMLPConfig:
    name : str = "MixedDeepMLP"
    time_embed_dim : int = 50

    continuous_embed_dim: int = 50
    discrete_embed_dim: int = 50
    merge_embed_dim: int = 100

    hidden_dim : int = 250
    activation : str = 'ReLU'
    num_layers : int = 4
    ema_decay: float = 0.999
    dropout : float = 0.2

    number_string:str = "num_nodes"

@dataclass
class MixedEGNN_dynamics_QM9Config:
    name : str = "MixedEGNN_dynamics_QM9"
    discrete_emb_dim:int = 5

    nf: int = 128
    n_layers: int = 6
    attention: bool = True
    tanh: bool = True
    inv_sublayers: int = 1
    norm_constant: float = 1
    sin_embedding: bool = False
    normalization_factor: float = 1
    aggregation_method: str = 'sum'
    mode:str='egnn_dynamics'

    context_node_nf:int = None
    conditioning: List[str] = None    # 'multiple arguments can be passed including: homo | onehot | lumo | num_atoms | etc. '
                                     # usage: "conditioning=['H_thermo', 'homo', 'onehot', 'H_thermo']"
    number_string:str = "num_atoms"
    ema_decay: float = 0.999

    """
    in_node_nf, context_node_nf,
    n_dims, hidden_nf=64, device='cpu',
    act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
    condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
    inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'
    """