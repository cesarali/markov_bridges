import lightning as L
from markov_bridges.data.dataloaders_utils import get_dataloaders
from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.configs.config_classes.trainers.trainer_config import CJBTrainerConfig
from markov_bridges.models.trainers.cjb_trainer import CJBTrainer
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig

from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.configs.config_classes.networks.temporal_networks_config import SequenceTransformerConfig

from markov_bridges.models.generative_models.cjb_lightning import ClassificationForwardRateL

if __name__=="__main__":
    train = True
    if train:
        experiment_config = CJBConfig()
        experiment_config.data = LakhPianoRollConfig(has_context_discrete=True,
                                                     context_discrete_dimension=64) # CHANGE
        # temporal network
        experiment_config.temporal_network = SequenceTransformerConfig(num_heads=1,num_layers=1) #CHANGE
        experiment_config.trainer = CJBTrainerConfig(
            number_of_epochs=10,
        )
        experiment_files = ExperimentFiles(experiment_name="cjb",
                                           experiment_type="music",
                                           experiment_indentifier="lightning_test",
                                           delete=True)
        experiment_files.create_directories(experiment_config)
        dataloaders = get_dataloaders(experiment_config)
        mixed_model = ClassificationForwardRateL(experiment_config)
        # saves checkpoints to 'some/path/' at every epoch end
        trainer = L.Trainer(default_root_dir=experiment_files.experiment_dir,
                            max_epochs=experiment_config.trainer.number_of_epochs)
        trainer.fit(mixed_model, dataloaders.train_dataloader, dataloaders.test_dataloader)