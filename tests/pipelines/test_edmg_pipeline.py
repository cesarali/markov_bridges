import torch
from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config

from markov_bridges.configs.config_classes.generative_models.edmg_config import (
    EDMGConfig,
    NoisingModelConfig
)
from markov_bridges.models.generative_models.edmg_lightning import EDGML
from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.utils.paralellism import check_model_devices

if __name__=="__main__":
    config = EDMGConfig()
    config.data = QM9Config(num_pts_train=1000,
                            num_pts_test=200,
                            num_pts_valid=200)    
    config.noising_model = NoisingModelConfig(n_layers=2)
    # conditioning=['H_thermo', 'homo']

    config.trainer.metrics = []
    experiment_files = ExperimentFiles(experiment_name="cjb",
                                       experiment_type="graph",
                                       experiment_indentifier="lightning_test9",
                                       delete=True)
    edmg = EDGML(config)

    one_hot, charges, x = edmg.pipeline.sample_chain(n_tries=1)
    
