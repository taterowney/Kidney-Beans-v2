#!/bin/bash

#SBATCH --job-name=training_data_thoughts
#SBATCH --partition=general
#SBATCH --output=logs/training_data_thoughts.out
#SBATCH --error=logs/training_data_thoughts.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=150G


# --exclude=babel-0-31

source ~/miniconda3/bin/activate KidneyBeans
export HF_HOME=/data/user_data/$USER/HF
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd ~/Kidney-Beans-v2
python eval.py
