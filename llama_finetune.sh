#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=4000

module load any/python/3.8.3-conda

conda activate nlp

ROOT=/gpfs/space/projects/stud_ml_22/NLP
RUN_NAME=tesla_with_8bit

nvidia-smi

gcc --version

python3.10 llama_finetune.py --output_dir $ROOT/experiments/$RUN_NAME --run_name $RUN_NAME



#sacct -j 42238848 --format=Elapsed

