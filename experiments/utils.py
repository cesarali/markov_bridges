
from markov_bridges.models.deprecated.trainers.cjb_trainer import CJBTrainer
from markov_bridges.utils.experiment_files import ExperimentFiles

def start_new_experiment(config, experiment_name="cjb", experiment_type="music"):
    experiment_files = ExperimentFiles(experiment_name, experiment_type)    
    trainer = CJBTrainer(config=config, experiment_files=experiment_files)
    trainer.train()

def train_from_last_checkpoint(path, experiment_name="cjb", experiment_type="music"):
    experiment_files = ExperimentFiles(experiment_name, experiment_type)    
    trainer = CJBTrainer(experiment_files=experiment_files,
                         experiment_dir=path,
                         starting_type="last") 
    trainer.train()
