#!/bin/bash -l
#SBATCH --job-name=robotwin-gt-probe
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/jiuquyun/home/Frank/logs/%x-%j.out
set -euo pipefail
cd /gpfs/jiuquyun/projects/Frank/FastWAM-stereotok-replay
test -z "$(git status --porcelain)"
git rev-parse HEAD
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/gpfs/jiuquyun/projects/Frank/fastwam-stereotok-tests/20261008/render-extras
export OMP_NUM_THREADS=1
srun /gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/train/bin/python tests/h100_robotwin_gt_probe.py
