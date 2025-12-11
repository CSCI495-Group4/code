#!/bin/bash
#SBATCH --job-name=modelA_train
#SBATCH --account=zheng826
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=modelA_train.out
#SBATCH --error=modelA_train.err

module load anaconda/2024.10-py312
source activate pytorch_env

cd /scratch/gilbreth/ichaudha/Phase4

python train_eval_modelA.py
