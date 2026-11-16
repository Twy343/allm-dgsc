#!/bin/bash
#SBATCH --job-name=allm-sc-once
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=h012
#SBATCH --constraint=R6000
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=disable-binding
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-sc-once-%j.log
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
source /insomnia001/depts/edu/users/yc4953/projects/allm-dgsc/scripts/activate_allm.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export JAX_COMPILATION_CACHE_DIR=/insomnia001/depts/edu/users/yc4953/hf_cache/jax_sc_test256
destination=outputs/sc-test256-once-v3
mkdir -p "$destination"
nvidia-smi > "$destination/gpu-${SLURM_JOB_ID}.txt"
python -m pip freeze > "$destination/packages-${SLURM_JOB_ID}.txt"
python -u -m exp_code.baselines.profile_then_run \
  --data-root /insomnia001/depts/edu/users/yc4953/datasets \
  --checkpoint /insomnia001/depts/edu/users/yc4953/models/diffusiongemma-26B-A4B-it \
  --output "$destination"
python -m exp_code.baselines.plot_benchmark_entropy --output "$destination"
