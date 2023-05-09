# AI-powered QA system for University of Tartu courses

## LLaMA inference on HPC

### Create conda environment
```
git clone https://github.com/DzvinkaYarish/tartu-nlp-courses-qa.git
conda create -n nlp python=3.10 --file tartu-nlp-courses-qa/nlp_env.txt
conda activate nlp

pip install sentencepiece
pip install accelerate
pip install git+https://github.com/huggingface/peft.git

pip install wandb # optional, for logging training to W&B
wandb login
```
### Clone acceleration library for Transformers
```
git clone https://github.com/timdettmers/bitsandbytes.git
cp tartu-nlp-courses-qa/Makefile bitsandbytes/
```

### Start interactive session on HPC

Optional: start srun in `tmux` to keep the session running after disconnecting from the server.

For tesla GPU:
```
srun --partition=gpu --gres=gpu:tesla:1  --mem=32G  --time=120 --cpus-per-task=4 --pty /bin/bash
```
For a100 GPU:
```
srun --partition=gpu --gres=gpu:a100-40g  --mem=32G  --time=120 --cpus-per-task=4 --pty /bin/bash
```

```
module load cuda/11.7.0 # for a100 gpu only!

module load any/python/3.8.3-conda

module load broadwell/gcc/5.2.0

conda activate nlp
```
Only when running inference for the first time:

```
cd bitsandbytes

CUDA_VERSION=117 make cuda11x

python3.10 setup.py install
```

#### Run LLaMA inference
```
cd ../tartu-nlp-courses-qa/
python3.10 llama_inference.py

```




