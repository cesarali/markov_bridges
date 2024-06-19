#!/bin/bash
#SBATCH --account=m4539              # PROJECT ID
#SBATCH --constraint=gpu 
#SBATCH --qos=regular
#SBATCH --job-name=music             
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=32           
#SBATCH --gpus-per-task=1             
#SBATCH --mem=16G                     # Real memory required
#SBATCH --time=24:00:00               # Total run time limit
#SBATCH --gres=gpu:1                                    
#SBATCH --output /global/homes/d/dfarough/markov_bridges/results/cjb/logs/%x_%j.log    
#SBATCH --error /global/homes/d/dfarough/markov_bridges/results/cjb/logs/%x_%j.err    
#SBATCH --mail-user darius.faroughy@rutgers.edu  

REPO_DIR="/global/homes/d/dfarough/markov_bridges"

echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Job Name: $SLURM_JOB_NAME"
echo "INFO: GPU stats"
nvidia-smi

module load conda
conda activate markov_bridges


cd $REPO_DIR
python experiments/music_experiment.py

#...end commands
            