import torch
import numpy as np
from typing import Union,List
from dataclasses import field,dataclass

from abc import ABC, abstractmethod
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.models.metrics.metrics_utils import LogMetrics

@dataclass
class TrainerState:
    model: Union[CJB]
    best_loss : float = np.inf

    average_train_loss : float = 0.
    average_test_loss : float = 0.

    test_loss: List[float] = field(default_factory=lambda:[])
    train_loss: List[float] = field(default_factory=lambda:[])

    epoch:int = 0
    number_of_test_step:int = 0
    number_of_training_steps:int = 0

    all_training_loss:List[float] = field(default_factory=lambda:[])

    def set_average_test_loss(self):
        if len(self.test_loss) > 0:
            self.average_test_loss = np.asarray(self.test_loss).mean()

    def set_average_train_loss(self):
        self.average_train_loss = np.asarray(self.train_loss).mean()

    def finish_epoch(self):
        self.test_loss = []
        self.train_loss = []
        
    def update_training_batch(self,loss):
        self.train_loss.append(loss)
        self.number_of_training_steps += 1
        self.all_training_loss.append(loss)

    def update_test_batch(self,loss):
        self.number_of_test_step += 1
        self.test_loss.append(loss)

# Assuming CTDDConfig, CTDD, and other necessary classes are defined elsewhere
class Trainer(ABC):
    """
    This trainer is intended to obtain a backward process of a markov jump via
    a ratio estimator with a stein estimator
    """

    dataloader = None
    generative_model:Union[CJB] = None
    config:Union[CJBConfig] = None
    do_ema:bool = False

    def parameters_info(self):
        print("# ==================================================")
        print("# START OF TRAINING ")
        print("# ==================================================")

        print("# Current Model ************************************")

        print(self.generative_model.experiment_files.experiment_type)
        print(self.generative_model.experiment_files.experiment_name)
        print(self.generative_model.experiment_files.experiment_indentifier)

        print("# ==================================================")
        print("# Number of Epochs {0}".format(self.number_of_epochs))
        print("# ==================================================")

    @abstractmethod
    def initialize(self):
        """
        Initializes the training process.
        To be implemented by subclasses.
        """
        pass

    def initialize_(self):
        self.initialize()
        self.parameters_info()
        self.writer = SummaryWriter(self.generative_model.experiment_files.tensorboard_path)
        self.tqdm_object = tqdm(range(self.config.trainer.number_of_epochs))
        self.best_metric = np.inf
        self.dataloader = self.generative_model.dataloader

        self.log_metrics = LogMetrics(self.generative_model,
                                      metrics_configs_list=self.config.trainer.metrics,
                            debug=True)
    
    @abstractmethod
    def train_step(self, current_model, databatch, number_of_training_step):
        """
        Defines a single training step.
        To be implemented by subclasses.
        """
        pass

    def global_training(self,training_state,all_metrics,epoch):
        return {},all_metrics

    @abstractmethod
    def test_step(self, current_model, databatch, number_of_test_step):
        """
        Defines a single test step.
        To be implemented by subclasses.
        """
        pass

    def global_test(self,training_state,all_metrics,epoch):
        return {},all_metrics

    @abstractmethod
    def preprocess_data(self, databatch):
        """
        Preprocesses the data batch.
        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def paralellize_model(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    def train(self):
        """
        FORWARD  means sampling from p_0 (data) -> p_1 (target)

        :return:
        """
        # INITIATE LOSS
        self.initialize_()
        all_metrics = {}
        results_ = {}
        self.saved = False

        training_state = TrainerState(self.generative_model,
                                      epoch=self.generative_model.config.trainer.epoch,
                                      number_of_training_steps=self.generative_model.config.trainer.number_of_training_steps,
                                      number_of_test_step=self.generative_model.config.trainer.number_of_test_step)
        
        if self.config.trainer.paralellize_gpu:
            self.paralellize_model()
            
        for epoch in self.tqdm_object:
            training_state.epoch = training_state.epoch + epoch
            #TRAINING
            for step, databatch in enumerate(self.dataloader.train()):
                databatch = self.preprocess_data(databatch)
                # DATA
                loss = self.train_step(databatch,training_state.number_of_training_steps,epoch)
                loss_ = loss.item() if isinstance(loss, torch.Tensor) else loss
                training_state.update_training_batch(loss_)
                self.tqdm_object.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
                self.tqdm_object.refresh()  # to show immediately the update
                if self.config.trainer.debug:
                    break
            training_state.set_average_train_loss()
            results_,all_metrics = self.global_training(training_state,all_metrics,epoch)

            #EVALUATES VALIDATION LOSS
            if not self.config.trainer.save_model_metrics_stopping:
                for step, databatch in enumerate(self.dataloader.test()):
                    databatch = self.preprocess_data(databatch)
                    # DATA
                    loss = self.test_step(databatch,training_state.number_of_test_step,epoch)
                    loss_ = loss.item() if isinstance(loss, torch.Tensor) else loss
                    training_state.update_test_batch(loss_)
                    if self.config.trainer.debug:
                        break

            # EVALUATES METRICS IF REQUIERED FOR STOPPING CRITERIA
            if self.config.trainer.save_model_metrics_stopping:
                if epoch > self.config.trainer.save_model_metrics_warming:
                    all_metrics = self.log_metrics(self.generative_model,epoch)
                    
            training_state.set_average_test_loss()
            results_,all_metrics = self.global_test(training_state,all_metrics,epoch)

            # STORING MODEL CHECKPOINTS
            if (epoch + 1) % self.config.trainer.save_model_epochs == 0:
                results_ = self.save_results(training_state,epoch+1,checkpoint=True,last_model=False)
            
            #STORE LAST MODEL
            results_ = self.save_results(training_state,epoch+1,checkpoint=False,last_model=True)

            # SAVE RESULTS IF LOSS DECREASES IN VALIDATION NOT BY IMPROVED METRICS
            if not self.config.trainer.save_model_metrics_stopping:
                current_average = training_state.average_test_loss if self.config.trainer.save_model_test_stopping else training_state.average_train_loss
                if current_average < training_state.best_loss:
                    if self.config.trainer.warm_up_best_model_epoch < epoch or epoch == self.number_of_epochs - 1:
                        results_ = self.save_results(training_state,epoch + 1,checkpoint=False,last_model=False)
                    training_state.best_loss = training_state.average_test_loss

            #SAVE RESULTS IF IT IMPROVES METRICS
            else:
                if epoch > self.config.trainer.save_model_metrics_warming:
                    if all_metrics[self.config.trainer.metric_to_save] < self.best_metric:
                        results_ = self.save_results(training_state, epoch + 1, checkpoint=False,last_model=False)
                        self.best_metric = all_metrics[self.config.trainer.metric_to_save]
            training_state.finish_epoch()

        #=====================================================
        # BEST MODEL IS READ AND METRICS ARE STORED
        #=====================================================
        experiment_dir = self.generative_model.experiment_files.experiment_dir
        if self.saved:
            self.generative_model = self.generative_model_class(experiment_dir=experiment_dir)

        all_metrics = self.log_metrics(self.generative_model,"best")
        self.writer.close()

        return results_,all_metrics

    def save_results(self,
                     training_state:TrainerState,
                     epoch:int,
                     checkpoint:bool=True,
                     last_model:bool=False):
        RESULTS = {
            "model": self.get_model(),
            "best_loss": training_state.best_loss,
            "training_loss":training_state.average_train_loss,
            "test_loss":training_state.average_test_loss,
            "all_training_loss":training_state.all_training_loss,
            "number_of_test_step":training_state.number_of_test_step,
            "number_of_training_steps":training_state.number_of_training_steps,
            "epoch":training_state.epoch,
        }
        if checkpoint:
            best_model_path_checkpoint = self.generative_model.experiment_files.best_model_path_checkpoint.format(epoch)
            torch.save(RESULTS,best_model_path_checkpoint)
        if last_model:
            best_model_path_checkpoint = self.generative_model.experiment_files.last_model
            torch.save(RESULTS,best_model_path_checkpoint)
        else:
            torch.save(RESULTS, self.generative_model.experiment_files.best_model_path)
        self.saved = True
        return RESULTS

