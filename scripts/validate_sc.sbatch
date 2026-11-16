#!/bin/bash
#SBATCH --job-name=allm-sc-validation
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-sc-validation-%j.log
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
root=/insomnia001/depts/edu/users/yc4953
source "$root/projects/allm-dgsc/scripts/activate_allm.sh"
export JAX_PLATFORMS=cpu
python -m unittest discover -s exp_code/tests -v
python -c 'import matplotlib; print("MATPLOTLIB", matplotlib.__version__)'
python -m exp_code.baselines.verify_tiny
