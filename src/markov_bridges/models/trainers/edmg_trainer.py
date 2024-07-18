import numpy as np
import torch
from tqdm import tqdm
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter

from typing import List
from dataclasses import dataclass,field
from markov_bridges.utils.experiment_files import ExperimentFiles

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR


from markov_bridges.models.generative_models.edmg import EDMG

import torch
import numpy as np
from torch.optim.adam import Adam
from markov_bridges.models.networks.utils.ema import EMA
from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMGConfig
from markov_bridges.models.trainers.abstract_trainer import TrainerState,Trainer
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from markov_bridges.utils.paralellism import nametuple_to_device
from markov_bridges.utils.equivariant_diffusion import (
    assert_mean_zero_with_mask, 
    remove_mean_with_mask,
    assert_correctly_masked, 
    sample_center_gravity_zero_gaussian_with_mask
)

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)
            
class EDMGTrainer(Trainer):

    config: EDMGConfig
    generative_model_class = EDMG
    name_ = "conditional_jump_bridge_trainer"

    def __init__(self,config=None,experiment_files=None,cjb=None,experiment_dir=None,starting_type="last"):
        """
        If experiment dir is provided, he loads the model from that folder and then creates
        a new folder 

        config: configuration file to start model
        cjb: model to train 
        experiment_files: files where to store the experiment
        experiment_dir: if provided experiment dir of model to load to continue training
        starting_type (str,int): for model in experiment_dir, defines which model to load, best, last or checkpoint if int provided

        if experiment_dir is provided, it will ignore config
        """
        if experiment_dir is not None:
            print("Starting Training from Model Provided in Experiment Dirs")
            self.generative_model = EDMG(experiment_dir=experiment_dir,type_of_load=starting_type)
            self.generative_model.experiment_files = experiment_files
            self.config = self.generative_model.config
            self.number_of_epochs = self.config.trainer.number_of_epochs
            device_str = self.config.trainer.device
            self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        else:
            self.config = config
            self.number_of_epochs = self.config.trainer.number_of_epochs
            device_str = self.config.trainer.device
            self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
            if cjb is None:
                self.generative_model = EDMG(self.config, experiment_files=experiment_files, device=self.device)
            else:
                self.generative_model = cjb
                self.dataloader = self.generative_model.dataloader

    def preprocess_data(self, databatch):    
        return databatch

    def paralellize_model(self):
        #DARIO
        self.generative_model.noising_model
    
    def get_model(self):
        return self.generative_model.forward_map.mixed_network

    def initialize(self):
        """
        Obtains initial loss to know when to save, restart the optimizer
        :return:
        """
        if isinstance(self.generative_model.noising_model,EMA) and self.config.trainer.do_ema:
            self.do_ema = True

        self.generative_model.start_new_experiment()

        #DEFINE OPTIMIZERS
        self.optimizer = Adam(self.generative_model.forward_map.parameters(),
                              lr=self.config.trainer.learning_rate,
                              weight_decay=self.config.trainer.weight_decay)
        
        self.lr = self.config.trainer.learning_rate

        if self.config.data.has_context_discrete:
            self.conditional_dimension = self.config.data.context_discrete_dimension
            self.generation_dimension = self.config.data.discrete_dimensions - self.conditional_dimension

        return np.inf

    def train_step(self,databatch:MarkovBridgeDataNameTuple, number_of_training_step,  epoch):
        #model_dp.train()
        #model.train()
        #nll_epoch = []
        #n_iterations = len(loader)

        x = databatch['positions'].to(self.device, self.dtype)
        node_mask = databatch['atom_mask'].to(self.device, self.dtype).unsqueeze(2)
        edge_mask = databatch['edge_mask'].to(self.device, self.dtype)
        one_hot = databatch['one_hot'].to(self.device, self.dtype)
        charges = (databatch['charges'] if args.include_charges else torch.zeros(0)).to(self.device, self.dtype)

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        self.optimizer.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z = self.generative_model.noising_model.loss(databatch)
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        self.optimizer.step()

        # Update EMA if enabled.
        #if args.ema_decay > 0:
        #    ema.update_model_average(model_ema, model)

        #if i % args.n_report_steps == 0:
        #    print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
        #          f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
        #          f"RegTerm: {reg_term.item():.1f}, "
        #          f"GradNorm: {grad_norm:.1f}")
            
        #nll_epoch.append(nll.item())
        #if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
        #    start = time.time()
        #    if len(args.conditioning) > 0:
        #        save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
        #    save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
        #                          batch_id=str(i))
        #    sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
        #                                     prop_dist, epoch=epoch)
        #    print(f'Sampling took {time.time() - start:.2f} seconds')

        #    vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
        #    vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
        #    if len(args.conditioning) > 0:
        #        vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
        #                            wandb=wandb, mode='conditional')

        return loss

    def test_step(self,databatch:MarkovBridgeDataNameTuple, number_of_test_step,epoch):
        with torch.no_grad():
            # gpu handling
            databatch = nametuple_to_device(databatch, self.device)

            # data pair and time sample
            discrete_sample,continuous_sample = self.generative_model.forward_map.sample_bridge(databatch)

            # sample x from z
            loss_ = self.generative_model.forward_map.loss(databatch,discrete_sample,continuous_sample)
        
            self.writer.add_scalar('test loss', loss_.item(), number_of_test_step)

        return loss_

            
