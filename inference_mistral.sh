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

source ~/miniconda3/bin/activate KidneyBeans
export HF_HOME=/data/user_data/$USER/HF
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve mistralai/Magistral-Small-2507 \
  --reasoning-parser mistral \
  --tokenizer_mode mistral \
  --config_format mistral \
  --load_format mistral \
  --tool-call-parser mistral \
  --enable-auto-tool-choice &

while ! nc -z localhost 8000; do   
  sleep 1 # wait for 1 second before checking again
done

echo "Server is up, starting eval script"
python3 eval_mistral.py
