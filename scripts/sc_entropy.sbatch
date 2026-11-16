#!/bin/bash
#SBATCH --job-name=allm-sc-entropy
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-sc-entropy-%j.log
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
source /insomnia001/depts/edu/users/yc4953/projects/allm-dgsc/scripts/activate_allm.sh
mkdir -p outputs/sc-entropy-v1
nvidia-smi > "outputs/sc-entropy-v1/gpu-${SLURM_JOB_ID}.txt"
python -m exp_code.baselines.entropy_sweep \
  --checkpoint /insomnia001/depts/edu/users/yc4953/models/diffusiongemma-26B-A4B-it \
  --output outputs/sc-entropy-v1
