#!/bin/bash

#SBATCH --job-name=AI_safety_classification
#SBATCH --partition=general
#SBATCH --output=logs/AI_safety_classification.out
#SBATCH --error=logs/AI_safety_classification.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=100G
#SBATCH --exclude=babel-15-36
#SBATCH --exclude=babel-15-32
#SBATCH --exclude=babel-1-27

source ~/miniconda3/bin/activate KidneyBeans
# source ~/miniconda3/bin/activate KidneyBeans2

export HF_HOME=/data/user_data/$USER/HF
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_DISABLE_XET=1 # why (why)
export VLLM_USE_FLASH_ATTENTION=0


cd ~/Kidney-Beans-v2
python eval.py
