from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.models.trainers.cjb_trainer import CJBTrainer

from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.networks.temporal_networks_config import SequenceTransformerConfig
from markov_bridges.configs.experiments_configs.music_experiments import conditional_music_experiment

def test_cjb_trainer():
    model_config = conditional_music_experiment()

    model_config.temporal_network = SequenceTransformerConfig(num_heads=1,num_layers=1)
    experiment_files = ExperimentFiles(experiment_name="cjb",experiment_type="tests")
    trainer = CJBTrainer(model_config,experiment_files)

    databatch = trainer.generative_model.dataloader.get_databatch()

    trainer.initialize_()
    trainer.train_step(databatch,1,1)

    return model_config 


if __name__=="__main__":
    test_cjb_trainer()