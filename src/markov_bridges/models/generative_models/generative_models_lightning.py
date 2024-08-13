import torch
import json
import lightning as L
from abc import ABC, abstractmethod
from lightning.pytorch.callbacks import ModelCheckpoint
from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataloader
from markov_bridges.data.qm9.qm9_points_dataloader import QM9PointDataloader
from markov_bridges.models.pipelines.abstract_pipeline import AbstractPipeline

from markov_bridges.configs.config_classes.generative_models.cfm_config import CFMConfig
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMGConfig

class AbstractGenerativeModelL(ABC):
    """
    """
    config_type:type
    config:CFMConfig|CJBConfig|CMBConfig|EDMGConfig = None
    experiment_files:ExperimentFiles=None
    model:L.LightningModule=None
    dataloader:MarkovBridgeDataloader|QM9PointDataloader=None
    pipeline:AbstractPipeline=None

    def __init__(
            self,
            config:CFMConfig|CJBConfig|CMBConfig|EDMGConfig=None,
            experiment_files:ExperimentFiles=None,
            experiment_source:str|ExperimentFiles=None,
            checkpoint_type:str="best"
        ):
        """
        """
        if experiment_files is not None:
            self.experiment_files = experiment_files
        elif config is not None:
            self.experiment_files = ExperimentFiles(experiment_name="generative_model",experiment_type="dummy",experiment_indentifier="dummy",delete=True)

        if config is not None:
            self.define_from_config(config)
            self.experiment_files.create_directories(config)
        elif experiment_source is not None:
            self.define_from_dir(experiment_source,checkpoint_type)

    def read_config(self,experiment_files:ExperimentFiles):
        config_json = json.load(open(experiment_files.config_path, "r"))
        if hasattr(config_json,"delete"):
            config_json["delete"] = False
        config = self.config_type(**config_json)   
        return config
    
    @abstractmethod
    def define_from_config(self,config):
        pass
    
    @abstractmethod
    def define_from_dir(self,experiment_dir=None,checkpoint_type:str="best"):
        pass

    #================================
    # TRAINING
    #================================
    @abstractmethod
    def test_evaluation(self)->dict:
        pass

    def get_trainer(self):
        checkpoint_callback_best = ModelCheckpoint(dirpath=self.experiment_files.experiment_dir, 
                                                   save_top_k=1, 
                                                   monitor="val_loss",
                                                   filename="best-{epoch:02d}")
        
        checkpoint_callback_last = ModelCheckpoint(dirpath=self.experiment_files.experiment_dir,
                                                   monitor="train_loss",
                                                   filename="last-{epoch:02d}")

        trainer = L.Trainer(default_root_dir=self.experiment_files.experiment_dir,
                            max_epochs=self.config.trainer.number_of_epochs,
                            callbacks=[checkpoint_callback_best,
                                       checkpoint_callback_last],
                            accelerator="auto",
                            devices=self.config.trainer.device)
        
        return trainer
    
    def train(self):
        trainer = self.get_trainer()
        trainer.fit(self.model, 
                    self.dataloader.train_dataloader, 
                    self.dataloader.validation_dataloader)
       
        all_metrics = self.test_evaluation() if len(self.config.trainer.metrics) else None

        return all_metrics