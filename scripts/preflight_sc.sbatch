#!/bin/bash
#SBATCH --job-name=allm-sc-inputs
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-sc-inputs-%j.log
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
source /insomnia001/depts/edu/users/yc4953/projects/allm-dgsc/scripts/activate_allm.sh
python -m unittest discover -s exp_code/tests -v
python -m exp_code.baselines.verify_inputs \
  --data-root /insomnia001/depts/edu/users/yc4953/datasets \
  --output outputs/sc-baseline-v1
