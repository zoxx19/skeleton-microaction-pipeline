#!/bin/bash
#SBATCH --job-name=mmn_train
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out

source /home/hpc/iwso/REDACTED_ACCOUNT/miniconda3/etc/profile.d/conda.sh
conda activate mmn_clean

cd ~/repos/Micro-Action/"Micro-action skeleton"/MMN
export PYTHONPATH=$PWD/torchlight:$PWD/torchpack:$PYTHONPATH

python scripts/main_single.py --config config/train/standing_single.yaml