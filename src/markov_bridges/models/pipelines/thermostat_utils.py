from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig

from markov_bridges.models.pipelines.thermostats import (
    ConstantThermostat,
    LogThermostat,
    ExponentialThermostat,
    InvertedExponentialThermostat
)

from markov_bridges.configs.config_classes.pipelines.cjb_thermostat_configs import (
  LogThermostatConfig,
  ExponentialThermostatConfig,
  InvertedExponentialThermostatConfig,
  ConstantThermostatConfig
)

def load_thermostat(config:CJBConfig):
  if isinstance(config.thermostat,ConstantThermostatConfig):
    thermostat = ConstantThermostat(config.thermostat)
  elif isinstance(config.thermostat,LogThermostatConfig):
    thermostat = LogThermostat(config.thermostat)
  elif isinstance(config.thermostat,ExponentialThermostatConfig):
    thermostat = ExponentialThermostat(config.thermostat)
  elif isinstance(config.thermostat, InvertedExponentialThermostatConfig):
    thermostat = InvertedExponentialThermostat(config.thermostat)
  else:
    raise Exception("No Thermostat Defined")
  return thermostat
