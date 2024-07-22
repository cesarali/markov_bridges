from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from markov_bridges.configs.config_classes.trainers.trainer_config import CMBTrainerConfig
from markov_bridges.configs.config_classes.networks.mixed_networks_config import MixedDeepMLPConfig
from markov_bridges.configs.config_classes.pipelines.pipeline_configs import BasicPipelineConfig
from markov_bridges.models.trainers.cmb_trainer import CMBTrainer
from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.configs.config_classes.metrics.metrics_configs import (
    MixedHellingerMetricConfig,
)

def get_independent_mix_experiment():
    model_config = CMBConfig()
    model_config.data = IndependentMixConfig(has_context_continuous=False,
                                             target_continuous_type="8gaussian",
                                             target_dirichlet=0.5)
    model_config.mixed_network = MixedDeepMLPConfig(num_layers=2,
                                                    hidden_dim=150,
                                                    time_embed_dim=50,
                                                    discrete_embed_dim=50)
    
    model_config.trainer = CMBTrainerConfig(number_of_epochs=20,
                                            debug=False,
                                            learning_rate=1e-4)
    model_config.trainer.metrics = []
    model_config.pipeline = BasicPipelineConfig(number_of_steps=200)
    return model_config

if __name__=="__main__":
    experiment_files = ExperimentFiles(experiment_name="cmb",
                                       experiment_type="independent") 
    model_config = get_independent_mix_experiment()
    trainer = CMBTrainer(config=model_config,
                         experiment_files=experiment_files)
    trainer.train()
    