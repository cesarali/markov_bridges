import lightning as L
from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.models.generative_models.cmb_lightning import MixedForwardMapL,CMBL
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from markov_bridges.configs.config_classes.trainers.trainer_config import CMBTrainerConfig
from markov_bridges.configs.config_classes.networks.mixed_networks_config import MixedDeepMLPConfig

from markov_bridges.configs.config_classes.metrics.metrics_configs import (
    MixedHellingerMetricConfig,
)
from markov_bridges.data.dataloaders_utils import get_dataloaders

if __name__=="__main__":
    train = True
    if train:
        model_config = CMBConfig(continuous_loss_type="flow",
                                 brownian_sigma=0.001)
        
        model_config.data = IndependentMixConfig(has_context_continuous=False,
                                                has_target_discrete=True,
                                                target_continuous_type="8gaussian",
                                                target_dirichlet=0.5,
                                                train_data_size=60000,
                                                test_data_size=10000)
        model_config.mixed_network = MixedDeepMLPConfig(num_layers=3,
                                                        hidden_dim=128,
                                                        time_embed_dim=16,
                                                        continuous_embed_dim=16,
                                                        discrete_embed_dim=16)
        model_config.trainer = CMBTrainerConfig(number_of_epochs=100,
                                                scheduler=None,
                                                warm_up=0,
                                                clip_grad=True,
                                                learning_rate=2e-4,
                                                metrics=[])
        model_config.trainer.metrics = [MixedHellingerMetricConfig(plot_continuous_variables=True,
                                                                   plot_histogram=True)]
        experiment_files = ExperimentFiles(experiment_name="cmb",
                                        experiment_type="independent",
                                        experiment_indentifier="lightning_test2",
                                        delete=True)
        cmb = CMBL(model_config,experiment_files)
        cmb.train()
    else:
        experiment_files = ExperimentFiles(experiment_name="cmb",experiment_type="independent",experiment_indentifier="lightning_test",delete=False)      
        cjb = CMBL(experiment_source=experiment_files,checkpoint_type="best")
        databatch = cjb.dataloader.get_databatch()