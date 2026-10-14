#!/bin/bash -l
#SBATCH --job-name=robotwin-gt-bench
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/jiuquyun/home/Frank/logs/%x-%j.out
set -euo pipefail
cd "${BENCH_REPO:-/gpfs/jiuquyun/projects/Frank/FastWAM-stereotok-replay}"
test -z "$(git status --porcelain)"
git rev-parse HEAD
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/gpfs/jiuquyun/projects/Frank/fastwam-stereotok-tests/20261008/render-extras
export PATH=/gpfs/jiuquyun/containers/tools/cuda-12.8/bin:$PATH
export CUDA_HOME=/gpfs/jiuquyun/containers/tools/cuda-12.8
srun --ntasks=1 /gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/train/bin/python -m unittest discover -s tests -p test_h100_robotwin_gt.py
srun --gpus-per-task=1 --kill-on-bad-exit=1 /gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/train/bin/python tests/benchmark_robotwin_gt.py
