import math
import torch
from torchtyping import TensorType

from markov_bridges.configs.config_classes.pipelines.cjb_thermostat_configs import (
    LogThermostatConfig,
    ExponentialThermostatConfig,
    InvertedExponentialThermostatConfig,
    ConstantThermostatConfig
)

from markov_bridges.utils.numerics.integration import integrate_quad_tensor_vec

class Thermostat:

    def __init__(self) -> None:
        return None
    
    def beta_integral(self, t1, t0):
        """
        Dummy integral for constant rate
        """
        if isinstance(self,ConstantThermostat):
            integral = (t1 - t0)*self.gamma
        else:
            integral = integrate_quad_tensor_vec(self.__call__, t0, t1, 100)
        return integral


class ConstantThermostat(Thermostat):

    def __init__(self,config:ConstantThermostatConfig):
        super().__init__()
        self.gamma = config.gamma

    def __call__(self, t):
        device = t.device
        thermostat = torch.full_like(t,self.gamma).to(device)
        return thermostat

    def integral(self,t0,t1):
        interval = t1 - t0
        integral = self.gamma * interval
        return integral

class LogThermostat(Thermostat):

    def __init__(self,config:LogThermostatConfig):
        super().__init__()
        self.time_base = config.time_base
        self.time_exponential = config.time_exponential

    def _integral_rate_scalar(self, t: TensorType["B"]) -> TensorType["B"]:
        integral_ = self.time_base * (self.time_exponential ** t) - self.time_base
        return integral_

    def __call__(self, t: TensorType["B"]) -> TensorType["B"]:
        device = t.device
        thermostat = self.time_base * math.log(self.time_exponential)* (self.time_exponential ** (1.- t))
        return thermostat.to(device)


class ExponentialThermostat(Thermostat):

    def __init__(self,config:ExponentialThermostatConfig):
        super().__init__()
        self.max = config.max
        self.gamma = config.gamma

    def _integral_rate_scalar(self, t):
        raise Exception

    def __call__(self, t):
        device = t.device
        thermostat = torch.exp(-self.gamma*torch.abs(t-0.5))*self.max
        return thermostat.to(device)

class InvertedExponentialThermostat(Thermostat):

    def __init__(self,config:InvertedExponentialThermostatConfig):
        super().__init__()
        self.max = config.max
        self.gamma = config.gamma

    def _integral_rate_scalar(self, t):
        raise Exception

    def __call__(self, t):
        device = t.device
        thermostat = torch.exp(-self.gamma*(t-0.5)) + torch.exp(self.gamma*(t-0.5))
        thermostat = thermostat/torch.exp(-self.gamma*(-torch.Tensor([0.5]))) + torch.exp(self.gamma*(-torch.Tensor([0.5])))
        thermostat = thermostat*self.max
        return thermostat.to(device)