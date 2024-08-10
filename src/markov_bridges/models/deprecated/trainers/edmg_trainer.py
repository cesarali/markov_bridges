import numpy as np
import torch
from tqdm import tqdm
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter

import markov_bridges.data.qm9.utils as qm9utils
from markov_bridges.models.generative_models.edmg import EDMG

import torch
import numpy as np
from torch.optim.adam import Adam
from markov_bridges.models.networks.utils.ema import EMA
from markov_bridges.configs.config_classes.generative_models.edmg_config import EDMGConfig
from markov_bridges.models.deprecated.trainers.abstract_trainer import TrainerState,Trainer
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from markov_bridges.utils.paralellism import nametuple_to_device
from markov_bridges.utils.equivariant_diffusion import (
    assert_mean_zero_with_mask, 
    remove_mean_with_mask,
    assert_correctly_masked, 
    sample_center_gravity_zero_gaussian_with_mask,
    random_rotation,
    gradient_clipping,
    Queue
)

from markov_bridges.data.qm9.utils import prepare_context, compute_mean_mad

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)
            
class EDMGTrainer(Trainer):

    config: EDMGConfig
    generative_model_class = EDMG
    name_ = "conditional_jump_bridge_trainer"

    def __init__(self,config=None,experiment_files=None,edmg=None,experiment_dir=None,starting_type="last"):
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
            self.noising_model_config = self.config.noising_model
            self.data_config = self.config.data
            self.number_of_epochs = self.config.trainer.number_of_epochs
            device_str = self.config.trainer.device
            self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        else:
            self.config = config
            self.noising_model_config = self.config.noising_model
            self.data_config = self.config.data
            self.number_of_epochs = self.config.trainer.number_of_epochs
            device_str = self.config.trainer.device
            self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
            if edmg is None:
                self.generative_model = EDMG(self.config, experiment_files=experiment_files, device=self.device)
            else:
                self.generative_model = edmg
                self.dataloader = self.generative_model.dataloader
        self.dtype = torch.float32
        
    def preprocess_data(self, databatch):    
        return databatch

    def paralellize_model(self):
        #DARIO
        self.generative_model.noising_model
    
    def get_model(self):
        return self.generative_model.noising_model

    def initialize(self):
        """
        Obtains initial loss to know when to save, restart the optimizer
        :return:
        """
        if isinstance(self.generative_model.noising_model,EMA) and self.config.trainer.do_ema:
            self.do_ema = True

        self.generative_model.start_new_experiment()
        #DEFINE OPTIMIZERS
        self.optimizer = Adam(self.generative_model.noising_model.parameters(),
                              lr=self.config.trainer.learning_rate,
                              amsgrad=self.config.trainer.amsgrad,
                              weight_decay=self.config.trainer.weight_decay)
        
        self.lr = self.config.trainer.learning_rate

        #==================================================
        #data_dummy = next(self.dataloader.train().__iter__())

        if len(self.noising_model_config.conditioning) > 0:
            print(f'Conditioning on {self.noising_model_config.conditioning}')
            self.property_norms = compute_mean_mad(self.dataloader, 
                                                   self.noising_model_config.conditioning, 
                                                   self.data_config.dataset)
            #context_dummy = prepare_context(self.noising_model_config.conditioning, data_dummy, property_norms)
            #context_node_nf = context_dummy.size(2)
        else:
            context_node_nf = 0
            property_norms = None

        self.config.noising_model.context_node_nf = context_node_nf

        self.gradnorm_queue = Queue()
        self.gradnorm_queue.add(3000)  # Add large value that will be flushed.
        
        return np.inf

    def train_step(self,databatch:MarkovBridgeDataNameTuple, number_of_training_step,  epoch):
        self.generative_model.noising_model.train()

        x = databatch['positions'].to(self.device, self.dtype)
        node_mask = databatch['atom_mask'].to(self.device, self.dtype).unsqueeze(2)
        edge_mask = databatch['edge_mask'].to(self.device, self.dtype)
        one_hot = databatch['one_hot'].to(self.device, self.dtype)
        charges = (databatch['charges'] if self.data_config.include_charges else torch.zeros(0)).to(self.device, self.dtype)

        # add noise 
        x = remove_mean_with_mask(x, node_mask)
        if self.noising_model_config.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * self.noising_model_config.augment_noise
        x = remove_mean_with_mask(x, node_mask)
        if self.noising_model_config.data_augmentation:
            x = random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(self.noising_model_config.conditioning) > 0:
            context = qm9utils.prepare_context(self.noising_model_config.conditioning, 
                                               databatch, 
                                               self.property_norms).to(self.device, self.dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        self.optimizer.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z = self.generative_model.noising_model.loss(x, 
                                                                             h, 
                                                                             node_mask, 
                                                                             edge_mask, 
                                                                             context,
                                                                             self.generative_model.nodes_dist)
        
        # standard nll from forward KL
        loss = nll + self.noising_model_config.ode_regularization * reg_term
        loss.backward()

        if self.config.trainer.clip_grad:
            grad_norm = gradient_clipping(self.generative_model.noising_model, 
                                          self.gradnorm_queue)
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

        return nll

    def test_step(self,databatch:MarkovBridgeDataNameTuple, number_of_test_step,epoch):
        self.generative_model.noising_model.eval()
        with torch.no_grad():
            x = databatch['positions'].to(self.device, self.dtype)
            node_mask = databatch['atom_mask'].to(self.device, self.dtype).unsqueeze(2)
            edge_mask = databatch['edge_mask'].to(self.device, self.dtype)
            one_hot = databatch['one_hot'].to(self.device, self.dtype)
            charges = (databatch['charges'] if self.data_config.include_charges else torch.zeros(0)).to(self.device, self.dtype)

            if self.noising_model_config.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * self.noising_model_config.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(self.noising_model_config.conditioning) > 0:
                context = qm9utils.prepare_context(self.noising_model_config.conditioning, 
                                                   databatch, 
                                                   self.property_norms).to(self.device, self.dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _ = self.generative_model.noising_model.loss(x,
                                                                 h, 
                                                                 node_mask, 
                                                                 edge_mask, 
                                                                 context,
                                                                 self.generative_model.nodes_dist)

        return nll

        

            
