#!/bin/bash
#SBATCH --job-name=manet_eval
#SBATCH --partition=work
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

source ~/miniconda3/bin/activate
conda activate openmmlab

cd ~/repos/Micro-Action/mar_scripts/manet/mmaction2

python tools/test.py \
  configs/recognition/manet/manet.py \
  checkpoints/best_top1_acc_epoch_40.pth \
  --out online_evaluation/test_results.pkl

python online_evaluation/eval.py
