from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from markov_bridges.configs.config_classes.trainers.trainer_config import CMBTrainerConfig

from markov_bridges.models.generative_models.cmb import CMB
from markov_bridges.models.trainers.cmb_trainer import CMBTrainer
from markov_bridges.utils.experiment_files import ExperimentFiles

import torch

def get_independent_mix_experiment():
    model_config = CMBConfig()
    model_config.data = IndependentMixConfig(has_context_discrete=True)
    model_config.trainer = CMBTrainerConfig(number_of_epochs=10,debug=True)
    model_config.trainer.metrics = []
    return model_config

if __name__=="__main__":
    experiment_files = ExperimentFiles(experiment_name="cmb",
                                       experiment_type="indepent") 
    model_config = get_independent_mix_experiment()
    trainer = CMBTrainer(config=model_config,experiment_files=experiment_files)
    trainer.train()
    