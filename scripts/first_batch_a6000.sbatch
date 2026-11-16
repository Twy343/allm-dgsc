#!/bin/bash
#SBATCH --job-name=allm-a6000-first-batch
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-a6000-first-batch-%j.log
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
source /insomnia001/depts/edu/users/yc4953/projects/allm-dgsc/scripts/activate_allm.sh
destination=outputs/sc-test256-a6000-v1
mkdir -p "$destination"
nvidia-smi > "$destination/gpu-${SLURM_JOB_ID}.txt"
nvidia-smi topo -m > "$destination/topology-${SLURM_JOB_ID}.txt"
trap 'status=$?; if [ "$status" -ne 0 ]; then scontrol release 13934635 || true; fi' EXIT
# First formal batch from the SAME fixed test set and output directory, not a pilot.
python -u -m exp_code.baselines.batched_run \
  --data-root /insomnia001/depts/edu/users/yc4953/datasets \
  --checkpoint /insomnia001/depts/edu/users/yc4953/models/diffusiongemma-26B-A4B-it \
  --output "$destination" --batch-size 4 --model-shards 2 --max-batches 1
