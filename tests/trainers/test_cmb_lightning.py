import lightning as L
from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.models.generative_models.cmb_lightning import MixedForwardMapL
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from markov_bridges.configs.config_classes.trainers.trainer_config import CMBTrainerConfig

from markov_bridges.data.dataloaders_utils import get_dataloaders


if __name__=="__main__":
    train = False
    if train:
        model_config = CMBConfig(continuous_loss_type="drift")
        model_config.data = IndependentMixConfig(has_context_discrete=True)
        model_config.trainer = CMBTrainerConfig(number_of_epochs=10,
                                                scheduler="exponential",
                                                warm_up=1,
                                                clip_grad=True)

        experiment_files = ExperimentFiles(experiment_name="cmb",
                                        experiment_type="independent",
                                        experiment_indentifier="lightning_test",
                                        delete=True)
        experiment_files.create_directories(model_config)
        dataloaders = get_dataloaders(model_config)
        mixed_model = MixedForwardMapL(model_config)
        # saves checkpoints to 'some/path/' at every epoch end
        trainer = L.Trainer(default_root_dir=experiment_files.experiment_dir,
                            max_epochs=model_config.trainer.number_of_epochs)
        trainer.fit(mixed_model, dataloaders.train_dataloader, dataloaders.test_dataloader)
    else:
        CKPT_PATH = r"C:\Users\cesar\Desktop\Projects\DiffusiveGenerativeModelling\OurCodes\markov_bridges\results\cmb\independent\lightning_test\lightning_logs\version_0\checkpoints\epoch=9-step=320.ckpt"
        model = MixedForwardMapL.load_from_checkpoint(CKPT_PATH)

        #checkpoint = torch.load(CKPT_PATH)
        #print(checkpoint.keys())