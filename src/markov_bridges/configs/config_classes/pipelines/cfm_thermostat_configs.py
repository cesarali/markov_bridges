from dataclasses import dataclass

@dataclass
class ConstantThermostatConfig:
    name:str="ConstantThermostat"
    gamma:float = 1e-3

