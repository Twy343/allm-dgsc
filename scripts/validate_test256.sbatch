#!/bin/bash
#SBATCH --job-name=allm-test256-inputs
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-test256-inputs-%j.log
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
source /insomnia001/depts/edu/users/yc4953/projects/allm-dgsc/scripts/activate_allm.sh
export JAX_PLATFORMS=cpu
python -m unittest discover -s exp_code/tests -v
python -u -m exp_code.baselines.verify_tiny
python -m exp_code.baselines.batched_run \
  --data-root /insomnia001/depts/edu/users/yc4953/datasets \
  --output outputs/sc-test256-v2 --batch-size 4 --prepare-only
