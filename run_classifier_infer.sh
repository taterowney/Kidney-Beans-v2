#!/bin/bash

#SBATCH --job-name=AI_safety_classification
#SBATCH --partition=general
#SBATCH --output=logs/AI_safety_classification.out
#SBATCH --error=logs/AI_safety_classification.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=100G
#SBATCH --exclude=babel-15-36
#SBATCH --exclude=babel-15-32

source ~/miniconda3/bin/activate KidneyBeans
export HF_HOME=/data/user_data/$USER/HF
export CUDA_VISIBLE_DEVICES=0,1

# export CUDA_LAUNCH_BLOCKING=1

cd ~/Kidney-Beans-v2
python classifier_infer.py
