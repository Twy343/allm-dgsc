#!/bin/bash
#SBATCH --job-name=allm-checkpoint
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-checkpoint-%j.log
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
root=/insomnia001/depts/edu/users/yc4953
source "$root/projects/allm-dgsc/scripts/activate_allm.sh"
if [ -s "$root/models/diffusiongemma-26B-A4B-it/verified-manifest.json" ]; then
  echo 'CHECKPOINT_ALREADY_VERIFIED'
  exit 0
fi
python -m exp_code.baselines.download_checkpoint --output "$root/models/diffusiongemma-26B-A4B-it"
