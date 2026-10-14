#!/bin/bash -l
#SBATCH --job-name=robotwin-gt
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=768G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/jiuquyun/home/Frank/logs/%x-%j.out
set -euo pipefail
repo=/gpfs/jiuquyun/projects/Frank/FastWAM-stereotok-replay
root=/gpfs/jiuquyun/projects/Frank/robotwin-stereo-gt-20261014
python=/gpfs/jiuquyun/projects/Frank/stereo-vae/runtime/train/bin/python
cd "$repo"
test -z "$(git status --porcelain)"
git rev-parse HEAD
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/gpfs/jiuquyun/projects/Frank/fastwam-stereotok-tests/20261008/render-extras
export PATH="$root/bin:/gpfs/jiuquyun/containers/tools/cuda-12.8/bin:$PATH"
export CUDA_HOME=/gpfs/jiuquyun/containers/tools/cuda-12.8
export TORCH_EXTENSIONS_DIR="$root/torch-extensions"
if [[ ${1:-} == prepare ]]; then
    srun --ntasks=1 "$python" scripts/h100_robotwin_gt.py prepare
elif [[ ${1:-} == smoke ]]; then
    srun --ntasks=1 --gpus-per-task=1 "$python" scripts/h100_robotwin_gt.py pool --smoke
else
    test -f "$root/episodes/0000/accepted.json"
    test -f "$root/episodes/0016/accepted.json"
    srun --gpu-bind=single:4 --kill-on-bad-exit=1 bash -c 'exec "$1" scripts/h100_robotwin_gt.py pool --worker "$SLURM_PROCID"' bash "$python"
    srun --ntasks=1 "$python" scripts/h100_robotwin_gt.py finalize
fi
