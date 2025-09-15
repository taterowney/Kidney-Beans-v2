#!/bin/bash

#SBATCH --job-name=training_data_thoughts
#SBATCH --partition=general
#SBATCH --output=logs/training_data_thoughts.out
#SBATCH --error=logs/training_data_thoughts.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=25G

source ~/miniconda3/bin/activate KidneyBeans
export HF_HOME=/data/user_data/$USER/HF

cd ~/Kidney-Beans-v2
python eval.py
