import lightning as L
from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.models.generative_models.cmb_lightning import MixedForwardMapL,CMBL
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from markov_bridges.configs.config_classes.data.graphs_configs import CommunitySmallGConfig
from markov_bridges.configs.config_classes.trainers.trainer_config import CMBTrainerConfig
from markov_bridges.configs.config_classes.networks.mixed_networks_config import MixedDeepMLPConfig

from markov_bridges.configs.config_classes.metrics.metrics_configs import (
    MixedHellingerMetricConfig,
)
from markov_bridges.data.dataloaders_utils import get_dataloaders
from markov_bridges.configs.config_classes.data.molecules_configs import QM9Config
from markov_bridges.configs.config_classes.networks.mixed_networks_config import MixedEGNN_dynamics_QM9Config

if __name__=="__main__":
    config = CMBConfig(continuous_loss_type="drift",
                       brownian_sigma=1.)
    config.data = QM9Config(num_pts_train=1000,
                            num_pts_test=200,
                            num_pts_valid=200,
                            include_charges=False)
    config.mixed_network = MixedEGNN_dynamics_QM9Config(n_layers=1,
                                                        conditioning=['H_thermo', 'homo'])
    config.trainer = CMBTrainerConfig(number_of_epochs=10,
                                      scheduler=None,
                                      warm_up=0,
                                      clip_grad=True,
                                      learning_rate=1e-4,
                                      metrics=[])
    experiment_files = ExperimentFiles(experiment_name="cmb",
                                        experiment_type="qm9",
                                        experiment_indentifier="lightning_test",
                                        delete=True)
    cmb = CMBL(config,experiment_files)
    cmb.train()
