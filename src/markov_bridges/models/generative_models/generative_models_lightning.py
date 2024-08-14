import torch
import json
from abc import ABC, abstractmethod
from dataclasses import asdict

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from markov_bridges.utils.experiment_files import ExperimentFiles
from markov_bridges.utils.training import EpochProgressBar

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
            # self.experiment_files.create_directories(config)
        elif experiment_source is not None:
            self.define_from_dir(experiment_source,checkpoint_type)

    def read_config(self,experiment_files:ExperimentFiles):
        config_json = json.load(open(experiment_files.config_path, "r"))
        if hasattr(config_json,"delete"):
            config_json["delete"] = False
        config = self.config_type(**config_json)   
        return config
    
    def save_config(self):
        if self.config is not None:
            config_as_dict = asdict(self.config)
            with open(self.experiment_files.config_path, "w") as file:
                json.dump(config_as_dict, file, indent=4)
    
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
        
        # Last epoch checkpoint
        checkpoint_callback_last = ModelCheckpoint(dirpath=self.experiment_files.experiment_dir,
                                                   save_top_k=1,
                                                   monitor=None,
                                                   filename="last-{epoch:02d}")
        
        progress_bar = EpochProgressBar()  # Use custom progress bar

        trainer = L.Trainer(default_root_dir=self.experiment_files.experiment_dir,
                            max_epochs=self.config.trainer.number_of_epochs,
                            callbacks=[progress_bar, 
                                       checkpoint_callback_best,
                                       checkpoint_callback_last, 
                                       ],
                            accelerator=self.config.trainer.accelerator,
                            devices=self.config.trainer.devices,
                            strategy=self.config.trainer.strategy)
        
        return trainer
    
    def train(self):
        trainer = self.get_trainer()
        trainer.fit(self.model, 
                    self.dataloader.train_dataloader, 
                    self.dataloader.validation_dataloader)
        
        self.save_config()
        all_metrics = self.test_evaluation() if len(self.config.trainer.metrics) else None

        return all_metrics