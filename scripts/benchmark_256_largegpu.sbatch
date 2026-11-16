#!/bin/bash
#SBATCH --job-name=allm-sc-test256-80gplus
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --constraint=h100|R6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-sc-test256-largegpu-%j.log
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
source /insomnia001/depts/edu/users/yc4953/projects/allm-dgsc/scripts/activate_allm.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
destination=outputs/sc-test256-largegpu-v1
mkdir -p "$destination"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$destination/gpu-${SLURM_JOB_ID}.txt"
# Check actual hardware; never infer VRAM from a scheduler alias or file size.
python -c 'import subprocess; sizes=subprocess.check_output(["nvidia-smi","--query-gpu=memory.total","--format=csv,noheader,nounits"],text=True).splitlines(); assert sizes and min(float(x) for x in sizes)>=75000, sizes'
python -m pip freeze > "$destination/packages-${SLURM_JOB_ID}.txt"
trap 'status=$?; if [ "$status" -ne 0 ]; then scontrol release 13934635 || true; fi' EXIT
python -u -m exp_code.baselines.batched_run \
  --data-root /insomnia001/depts/edu/users/yc4953/datasets \
  --checkpoint /insomnia001/depts/edu/users/yc4953/models/diffusiongemma-26B-A4B-it \
  --output "$destination" --batch-size 4 --model-shards 1
python -m exp_code.baselines.plot_benchmark_entropy --output "$destination"
scancel 13934635 || true
