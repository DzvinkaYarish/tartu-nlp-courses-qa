#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40g
#SBATCH --time=4000
####### --nodelist=falcon5

module load cuda/11.7.0
module load any/python/3.8.3-conda

conda activate nlp

ROOT=/gpfs/space/projects/stud_ml_22/NLP
RUN_NAME=a100_longer_training_vicuna

nvidia-smi

gcc --version

python3.10 llama_finetune.py --output_dir $ROOT/experiments/$RUN_NAME --run_name $RUN_NAME --bf16



#sacct -j 42238848 --format=Elapsed

