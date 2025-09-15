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

# source ~/miniconda3/bin/activate KidneyBeans
source ~/miniconda3/bin/activate KidneyBeans2

export HF_HOME=/data/user_data/$USER/HF
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_DISABLE_XET=1 # why (why)



cd ~/Kidney-Beans-v2
# python eval.py
# exit 0

echo "Starting vLLM server..."

export VLLM_DISABLE_COMPILE_CACHE=1

# vllm serve microsoft/Phi-4-reasoning \
#   --tensor-parallel-size 2 \
#   --pipeline-parallel-size 4 \
#   -O0 \
#   --enforce-eager &

# vllm serve mistralai/Magistral-Small-2507 \
#   --tensor-parallel-size 4 \
#   --pipeline-parallel-size 2 \
#   -O0 \
#   --reasoning-parser mistral --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice &

# exit 1

vllm serve mistralai/Ministral-8B-Instruct-2410 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  -O0 \
  --tokenizer_mode mistral --config_format mistral --load_format mistral &

# vllm serve meta-llama/Llama-Guard-3-8B \
#   --tensor-parallel-size 4 \
#   --pipeline-parallel-size 1 \
#   -O0 &

echo "Waiting..."
while ! nc -z localhost 8000; do
  sleep 1
done
echo "vLLM server is up!"

python eval.py