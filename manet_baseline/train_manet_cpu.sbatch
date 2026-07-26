#!/bin/bash
#SBATCH --job-name=manet_cpu
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/miniconda3/bin/activate
conda activate openmmlab

cd ~/repos/Micro-Action/mar_scripts/manet/mmaction2

hostname
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

python tools/train.py configs/recognition/manet/manet.py --work-dir work_dirs/manet_run1 --seed=0 --deterministic
