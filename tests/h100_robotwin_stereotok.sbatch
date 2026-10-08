#!/bin/bash -l
#SBATCH --job-name=fastwam-stereotok-smoke
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
set -euo pipefail
repo=$1
python=$2
checkpoint=$3
output=$4
extras=$5
cd "$repo"
test -z "$(git status --porcelain)"
git log -1 --format='%H %s'
export PYTHONPATH="$extras:$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$output/inductor"
export TRITON_CACHE_DIR="$output/triton"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
srun nvidia-smi
srun "$python" tests/test_robotwin_stereotok.py -v
if [ ! -f "$output/student-u8000.pt" ]; then
    srun "$python" scripts/export_stereotok_student.py --checkpoint "$checkpoint" \
        --output "$output/student-u8000.pt"
fi
srun "$python" tests/accept_robotwin_stereotok_gpu.py \
    --student "$output/student-u8000.pt" --report "$output/gpu-${SLURM_JOB_ID}.json"
