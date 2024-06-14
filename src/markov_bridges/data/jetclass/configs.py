import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

""" Default configurations for JetClass datasets.
"""

@dataclass
class JetClass_Config:
    NAME : str = None
    DATASET : str = 'jetclass'
    DATA_SOURCE : str = 'qcd'
    DATA_TARGET : str = 'top'
    MAX_NUM_CONSTITUENTS : int = 30
    FEATURES : str = 'constituents'
    PREPROCESS : List[str] = field(default_factory = lambda : ['standardize'])
    DIM_INPUT : int = 3

