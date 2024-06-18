from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.networks.temporal_networks_config import SequenceTransformerConfig
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.metrics.metrics_configs import MetricsAvaliable, HellingerMetricConfig

# models
from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.models.trainers.cjb_trainer import CJBTrainer
from markov_bridges.utils.experiment_files import ExperimentFiles


def conditional_music_experiment_config():
    config = CJBConfig()
    config.data = LakhPianoRollConfig(has_context_discrete=True)

    config.temporal_network = SequenceTransformerConfig(num_heads=1,num_layers=1) 

    config.pipeline.number_of_steps = 10 
    config.trainer.number_of_epochs = 3
    config.trainer.device = 'cuda:0' 
    config.trainer.warm_up = 5000  
    config.trainer.learning_rate = 2e-4
    config.trainer.metrics = [] #[HellingerMetricConfig()]
    return config


if __name__=="__main__":

    from utils import start_new_experiment, train_from_last_checkpoint

    config = conditional_music_experiment_config()
    start_new_experiment(config=config, experiment_name="cjb", experiment_type="music")

    # experiment_dir = r"/global/homes/d/dfarough/markov_bridges/results/cjb/music/1718717862"
    # train_from_last_checkpoint(path=experiment_dir)

