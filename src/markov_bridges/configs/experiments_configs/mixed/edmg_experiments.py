from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config

from markov_bridges.configs.config_classes.generative_models.edmg_config import (
    EDMGConfig,
    NoisingModelConfig
)

from markov_bridges.configs.config_classes.trainers.trainer_config import EDMGTrainerConfig
from markov_bridges.configs.config_classes.pipelines.pipeline_configs import BasicPipelineConfig

from markov_bridges.models.generative_models.edmg import EDMG
from markov_bridges.models.networks.temporal.edmg.edmg_utils import get_edmg_model
from markov_bridges.data.dataloaders_utils import get_dataloaders

from markov_bridges.models.deprecated.trainers.edmg_trainer import EDMGTrainer
from markov_bridges.utils.experiment_files import ExperimentFiles

from markov_bridges.configs.config_classes.metrics.metrics_configs import (
    MixedHellingerMetricConfig,
)

def get_independent_mix_experiment():
    model_config = EDMGConfig()
    model_config.data = QM9Config(num_pts_train=1000,
                                  num_pts_test=200,
                                  num_pts_valid=200)
    model_config.trainer = EDMGTrainerConfig(number_of_epochs=200,
                                             debug=True,
                                             learning_rate=1e-4)
    model_config.trainer.metrics = []
    model_config.pipeline = BasicPipelineConfig(number_of_steps=200)

    return model_config

if __name__=="__main__":
    experiment_files = ExperimentFiles(experiment_name="edmg",
                                       experiment_type="independent") 
    model_config = get_independent_mix_experiment()
    trainer = EDMGTrainer(config=model_config,
                          experiment_files=experiment_files)
    trainer.train()
    