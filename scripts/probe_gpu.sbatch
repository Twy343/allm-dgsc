#!/bin/bash
#SBATCH --job-name=allm-gpu-capacity
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:03:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-gpu-capacity-%j.log
set -euo pipefail
source /insomnia001/depts/edu/users/yc4953/projects/allm-dgsc/scripts/activate_allm.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c 'import jax,jax.numpy as j; print(jax.devices()); print(j.sum(j.ones((1024,1024))).block_until_ready())'
