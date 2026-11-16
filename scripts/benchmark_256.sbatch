#!/bin/bash
#SBATCH --job-name=allm-sc-test256
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-sc-test256-%j.log
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
source /insomnia001/depts/edu/users/yc4953/projects/allm-dgsc/scripts/activate_allm.sh
destination=outputs/sc-test256-v2
mkdir -p "$destination"
nvidia-smi > "$destination/gpu-${SLURM_JOB_ID}.txt"
python -m pip freeze > "$destination/packages-${SLURM_JOB_ID}.txt"
python -u -m exp_code.baselines.batched_run \
  --data-root /insomnia001/depts/edu/users/yc4953/datasets \
  --checkpoint /insomnia001/depts/edu/users/yc4953/models/diffusiongemma-26B-A4B-it \
  --output "$destination" --batch-size 4
python -m exp_code.baselines.plot_benchmark_entropy --output "$destination"
