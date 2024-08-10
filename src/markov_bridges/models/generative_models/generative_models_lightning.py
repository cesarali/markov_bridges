import torch
import lightning as L
from abc import ABC, abstractmethod
from lightning.pytorch.callbacks import ModelCheckpoint
from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from markov_bridges.models.pipelines.abstract_pipeline import AbstractPipeline

from markov_bridges.configs.config_classes.generative_models.cfm_config import CFMConfig
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMGConfig

class AbstractGenerativeModelL(ABC):
    """
    """
    config:CFMConfig|CJBConfig|CMBConfig|EDMGConfig = None
    experiment_files:ExperimentFiles=None
    model:L.LightningModule=None
    dataloader:MarkovBridgeDataloader=None
    pipeline:AbstractPipeline=None

    def __init__(self,config,experiment_files=None,experiment_dir=None,checkpoint_path=None):
        """
        """
        if self.experiment_files is not None:
            self.experiment_files = experiment_files
        else:
            self.experiment_files = ExperimentFiles(experiment_name="generative_model",
                                                    experiment_type="dummy",
                                                    experiment_indentifier="dummy",
                                                    delete=True)
        if config is not None:
            self.define_from_config(config)
            self.experiment_files.create_directories(config)
        elif experiment_dir is not None:
            self.define_from_dir(experiment_dir,checkpoint_path)
        
    @abstractmethod
    def define_from_config(self,config):
        pass
    
    @abstractmethod
    def define_from_dir(self,experiment_dir=None,checkpoint_dir=None):
        pass

    #================================
    # TRAINING
    #================================
    @abstractmethod
    def test_evaluation(self)->dict:
        pass

    def get_trainer(self):
        checkpoint_callback = ModelCheckpoint(dirpath=self.experiment_files.experiment_dir, 
                                              save_top_k=2, 
                                              monitor="val_loss")
        trainer = L.Trainer(default_root_dir=self.experiment_files.experiment_dir,
                            max_epochs=self.config.trainer.number_of_epochs,
                            callbacks=[checkpoint_callback])
        
        return trainer
    
    def train(self):
        trainer = self.get_trainer()
        trainer.fit(self.model, 
                    self.dataloader.train_dataloader, 
                    self.dataloader.validation_dataloader)
        all_metrics = self.test_evaluation()
        return all_metrics