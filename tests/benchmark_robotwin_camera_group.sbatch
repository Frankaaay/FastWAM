#!/bin/bash -l
#SBATCH --job-name=robotwin-camera-group
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/jiuquyun/home/Frank/logs/%x-%j.out
set -euo pipefail
cd /gpfs/jiuquyun/projects/Frank/FastWAM-stereotok-robotwin
test -z "$(git status --porcelain)"
git rev-parse HEAD
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/gpfs/jiuquyun/projects/Frank/fastwam-stereotok-tests/20261008/render-extras
srun /gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/train/bin/python tests/benchmark_robotwin_camera_group.py
