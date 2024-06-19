from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.networks.temporal_networks_config import SequenceTransformerConfig
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.metrics.metrics_configs import MetricsAvaliable, HellingerMetricConfig

# models
from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.models.trainers.cjb_trainer import CJBTrainer
from markov_bridges.utils.experiment_files import ExperimentFiles


def conditional_music_experiment_config(batch_size, gamma):
    config = CJBConfig()
    config.data = LakhPianoRollConfig(has_context_discrete=True)
    config.temporal_network = SequenceTransformerConfig(num_layers=6, num_heads=8) 
    config.pipeline.number_of_steps = 1000 
    config.trainer.number_of_epochs = 10000
    config.trainer.device = 'cuda:0' 
    config.trainer.warm_up = 5000  
    config.trainer.learning_rate = 2e-4
    config.trainer.clip_grad = True
    config.trainer.clip_max_norm=1.0
    config.trainer.metrics = [] #[HellingerMetricConfig()]
    config.data.batch_size = batch_size
    config.data.context_dimension =  128
    config.thermostat.gamma = gamma                                
    return config


if __name__=="__main__":

    from utils import start_new_experiment, train_from_last_checkpoint

    config = conditional_music_experiment_config(batch_size=128, gamma=1/129.)
    experiment_files = ExperimentFiles(experiment_type="piano_roll_transformer_10k_epochs_uniform",
                                       experiment_name="cjb",
                                       experiment_indentifier='gamma_optimal_128_context')  
      
    trainer = CJBTrainer(config=config, experiment_files=experiment_files)
    trainer.train()

    # experiment_dir = r"/global/homes/d/dfarough/markov_bridges/results/cjb/music/1718717862"
    # train_from_last_checkpoint(path=experiment_dir)

