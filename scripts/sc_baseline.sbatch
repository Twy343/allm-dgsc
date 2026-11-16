#!/bin/bash
#SBATCH --job-name=allm-sc-baseline
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-sc-baseline-%j.log
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
root=/insomnia001/depts/edu/users/yc4953
source "$root/projects/allm-dgsc/scripts/activate_allm.sh"
mkdir -p outputs/sc-baseline-v1
nvidia-smi > "outputs/sc-baseline-v1/gpu-${SLURM_JOB_ID}.txt"
python -m pip freeze > "outputs/sc-baseline-v1/packages-${SLURM_JOB_ID}.txt"
# Run a short checkpoint-backed check before the 64-per-task pilot.
python -m exp_code.baselines.run --data-root "$root/datasets" \
  --checkpoint "$root/models/diffusiongemma-26B-A4B-it" \
  --output outputs/sc-baseline-v1/smoke --stage smoke
python -m exp_code.baselines.run --data-root "$root/datasets" \
  --checkpoint "$root/models/diffusiongemma-26B-A4B-it" \
  --output outputs/sc-baseline-v1/pilot --stage pilot
