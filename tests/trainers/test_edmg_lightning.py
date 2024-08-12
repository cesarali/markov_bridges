from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config
from markov_bridges.configs.config_classes.generative_models.edmg_config import (
    EDMGConfig,
    NoisingModelConfig
)
from markov_bridges.configs.config_classes.trainers.trainer_config import EDMGTrainerConfig
from markov_bridges.models.generative_models.edmg_lightning import EDGML
from markov_bridges.utils.experiment_files import ExperimentFiles

if __name__=="__main__":
    model_config = EDMGConfig()
    model_config.data = QM9Config(num_pts_train=1000,
                                  num_pts_test=200,
                                  num_pts_valid=200)
    model_config.noising_model = NoisingModelConfig(conditioning=['H_thermo', 'homo' ])
    model_config.trainer = EDMGTrainerConfig(number_of_epochs=200,
                                             debug=True,
                                             learning_rate=1e-4)
    model_config.trainer.metrics = []
    experiment_files = ExperimentFiles(experiment_name="cjb",
                                       experiment_type="graph",
                                       experiment_indentifier="lightning_test9",
                                       delete=True)
    cjb = EDGML(model_config,experiment_files)
    cjb.train()