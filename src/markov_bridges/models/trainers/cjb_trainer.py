import numpy as np
import torch
from tqdm import tqdm
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter

from typing import List
from dataclasses import dataclass,field
from markov_bridges.utils.experiment_files import ExperimentFiles


from markov_bridges.models.generative_models.cjb import CJB

import torch
import numpy as np
from torch.optim.adam import Adam
from markov_bridges.models.networks.utils.ema import EMA
from markov_bridges.configs.config_classes.generative_models.cjb_config import CJBConfig
from markov_bridges.models.trainers.abstract_trainer import TrainerState,Trainer
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple

class CJBTrainer(Trainer):

    config: CJBConfig
    generative_model: CJB
    generative_model_class = CJB
    name_ = "conditional_jump_bridge_trainer"

    def __init__(self,config,experiment_files,crm=None):
        self.config = config
        self.number_of_epochs = self.config.trainer.number_of_epochs
        device_str = self.config.trainer.device

        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        if crm is None:
            self.generative_model = CJB(self.config, experiment_files=experiment_files, device=self.device)
        else:
            self.generative_model = crm
            self.dataloader = self.generative_model.dataloader

    def preprocess_data(self, databatch):
        return databatch

    def get_model(self):
        return self.generative_model.forward_rate

    def initialize(self):
        """
        Obtains initial loss to know when to save, restart the optimizer
        :return:
        """
        if isinstance(self.generative_model.forward_rate,EMA) and self.config.trainer.do_ema:
            self.do_ema = True

        self.generative_model.start_new_experiment()

        #DEFINE OPTIMIZERS
        self.optimizer = Adam(self.generative_model.forward_rate.parameters(),
                              lr=self.config.trainer.learning_rate,
                              weight_decay=self.config.trainer.weight_decay)
        
        self.lr = self.config.trainer.learning_rate
        self.scheduler = None

        self.loss_stats = {}
        self.loss_stats_variance = {}
        self.loss_variance_times = torch.linspace(0.001,1.,20)

        if self.config.trainer.lr_decay:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                    gamma=self.config.trainer.lr_decay)

        if self.config.data.has_context_discrete:
            self.conditional_dimension = self.config.data.context_dimension
            self.generation_dimension = self.config.data.dimensions - self.conditional_dimension

        return np.inf

    def train_step(self,databatch:MarkovBridgeDataNameTuple, number_of_training_step,epoch):
        conditional_dimension = self.config.data.context_dimension

        join_context = lambda context_discrete,data_discrete : torch.cat([context_discrete,data_discrete],dim=1)
        remove_context = lambda full_data_discrete : full_data_discrete[:,conditional_dimension:]

        # data pair and time sample
        target_discrete, source_discrete = self.generative_model.sample_pair(databatch,self.device)

        # time selection
        batch_size = source_discrete.size(0)
        time = torch.rand(batch_size).to(self.device)

        # sample x from z
        sampled_x = self.generative_model.forward_rate.sample_x(target_discrete, source_discrete, time).float()
            
        # loss
        if self.config.data.has_context_discrete:
            completed_sampled_x = join_context(databatch.context_discrete,sampled_x)
            model_classification = self.generative_model.forward_rate.classify(completed_sampled_x, time)
            model_classification_ = model_classification[:, self.config.data.context_dimension:,:]
            # reshape for cross logits
            model_classification_ = model_classification_.reshape(-1, self.config.data.vocab_size)
            target_discrete_ = databatch.target_discrete.reshape(-1)

            loss = self.generative_model.loss(model_classification_, target_discrete_.long())
        else:
            model_classification = self.generative_model.forward_rate.classify(sampled_x, time)
            model_classification_ = model_classification.view(-1, self.config.data1.vocab_size)
            x_1 = x_1.view(-1)
            loss = self.generative_model.loss(model_classification_,x_1.long())

        loss = loss.mean()
        # optimization
        self.optimizer.zero_grad()
        loss.backward()

        if self.config.trainer.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.generative_model.forward_rate.parameters(), self.config.trainer.clip_max_norm)

        if self.config.trainer.warm_up > 0:
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr * np.minimum(float(number_of_training_step) / self.config.trainer.warm_up, 1.0)

        self.optimizer.step()

        if self.do_ema:
            self.generative_model.forward_rate.update_ema()

        if self.config.trainer.lr_decay:
            self.scheduler.step()

        self.writer.add_scalar('training loss', loss.item(), number_of_training_step)
        return loss

    def test_step(self,databatch,number_of_test_step,epoch):
        batch_0, batch_1 = databatch
        with torch.no_grad():
            # data pair and time sample
            x_1, x_0 = self.generative_model.sample_pair(batch_1, batch_0, self.device)

            x_0 = x_0.float().to(self.device)
            x_1 = x_1.float().to(self.device)

            if len(x_0.shape) > 2:
                batch_size = x_0.size(0)
                x_0 = x_0.reshape(batch_size,-1)

            if len(x_1.shape) > 2:
                batch_size = x_1.size(0)
                x_1 = x_1.reshape(batch_size,-1)

            # time selection
            batch_size = x_0.size(0)
            time = torch.rand(batch_size).to(self.device)

            # sample x from z
            sampled_x = self.generative_model.forward_rate.sample_x(x_1, x_0, time)

            #LOSS
            model_classification = self.generative_model.forward_rate.classify(sampled_x.float(), time)
            model_classification_ = model_classification.view(-1, self.config.data1.vocab_size)
            x_1 = x_1.view(-1)
            loss = self.generative_model.loss(model_classification_,x_1.long())
            if self.config.trainer.loss_regularize:
                if self.config.trainer.loss_regularize_square:
                    rate_regularizer = self.generative_model.forward_rate.thermostat(time)
                else:
                    rate_regularizer = self.generative_model.forward_rate.thermostat(time)
                rate_regularizer = rate_regularizer[:, None]
                rate_regularizer = rate_regularizer.repeat((1, self.config.data1.dimensions)).view(-1)
                loss = rate_regularizer*loss
            loss = loss.mean()
            self.writer.add_scalar('test loss', loss.item(), number_of_test_step)

        return loss

    def global_test(self,training_state,all_metrics,epoch):
        if "loss_variance_times" in self.config.trainer.metrics:
            with torch.no_grad():
                loss_mean_per_time = []
                loss_variance_per_time = []
                for time_ in self.loss_variance_times:
                    # =================================
                    # LOSS PER TIME
                    # =================================
                    loss_batch = []
                    # VALIDATION
                    for step, databatch in enumerate(self.dataloader.test()):
                        databatch = self.preprocess_data(databatch)
                        batch_0, batch_1 = databatch
                        # data pair and time sample
                        x_1, x_0 = self.generative_model.sample_pair(batch_1, batch_0, self.device)

                        x_0 = x_0.float().to(self.device)
                        x_1 = x_1.float().to(self.device)

                        if len(x_0.shape) > 2:
                            batch_size = x_0.size(0)
                            x_0 = x_0.reshape(batch_size, -1)

                        if len(x_1.shape) > 2:
                            batch_size = x_1.size(0)
                            x_1 = x_1.reshape(batch_size, -1)

                        # time selection
                        batch_size = x_0.size(0)

                        time = torch.ones(batch_size).to(self.device)
                        time = time*time_
                        # sample x from z
                        sampled_x = self.generative_model.forward_rate.sample_x(x_1, x_0, time)

                        # LOSS
                        model_classification = self.generative_model.forward_rate.classify(x_1, time)
                        model_classification_ = model_classification.view(-1, self.config.data1.vocab_size)
                        sampled_x = sampled_x.view(-1)
                        loss = self.generative_model.loss(model_classification_, sampled_x)
                        loss_batch.extend(loss.detach().numpy().tolist())
                        if self.config.trainer.debug:
                            break

                    loss_batch_mean = np.asarray(loss_batch).mean()
                    loss_batch_variance = np.asarray(loss_batch).std()
                    loss_mean_per_time.append(loss_batch_mean)
                    loss_variance_per_time.append(loss_batch_variance)
                self.loss_stats[epoch] = loss_mean_per_time
                self.loss_stats_variance[epoch] = loss_variance_per_time

                all_metrics["loss_mean_times"] = self.loss_stats
                all_metrics["loss_variance_times"] = self.loss_stats_variance
        return {},all_metrics

