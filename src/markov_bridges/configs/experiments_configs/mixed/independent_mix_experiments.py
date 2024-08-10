from markov_bridges.models.deprecated.trainers.cmb_trainer import CMBTrainer

from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from markov_bridges.configs.config_classes.trainers.trainer_config import CMBTrainerConfig
from markov_bridges.configs.config_classes.pipelines.pipeline_configs import CMBPipelineConfig
from markov_bridges.configs.config_classes.networks.mixed_networks_config import MixedDeepMLPConfig

from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.configs.config_classes.metrics.metrics_configs import (
    MixedHellingerMetricConfig,
)
from markov_bridges.models.generative_models.cmb_lightning import CMBL

def get_independent_mix_experiment():
    model_config = CMBConfig(continuous_loss_type="flow",
                             brownian_sigma=1.)
    model_config.data = IndependentMixConfig(has_context_continuous=False,
                                             has_target_discrete=True,
                                             target_continuous_type="8gaussian",
                                             target_dirichlet=0.2,
                                             train_data_size=2000,
                                             test_data_size=1000)
    model_config.mixed_network = MixedDeepMLPConfig(num_layers=3,
                                                    hidden_dim=150,
                                                    time_embed_dim=50,
                                                    continuous_embed_dim=50,
                                                    discrete_embed_dim=20)
    model_config.trainer = CMBTrainerConfig(number_of_epochs=10,
                                            learning_rate=1e-4)
    model_config.trainer.metrics = [MixedHellingerMetricConfig(plot_continuous_variables=True,
                                                               plot_histogram=True)]
    #model_config.trainer.metrics = []
    model_config.pipeline = CMBPipelineConfig(number_of_steps=200,solver="ode_tau")
    return model_config

if __name__=="__main__":
    experiment_files = ExperimentFiles(experiment_name="cmb",
                                       experiment_type="independent",
                                       experiment_indentifier="old",
                                       delete=True) 
    model_config = get_independent_mix_experiment()
    # cmb = CMBL(model_config,experiment_files)
    # cmb.train()
    
    trainer = CMBTrainer(config=model_config,
                         experiment_files=experiment_files)
    trainer.train()
    