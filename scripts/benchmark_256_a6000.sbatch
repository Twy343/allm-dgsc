#!/bin/bash
#SBATCH --job-name=allm-sc-test256-a6000
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-sc-test256-a6000-%j.log
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
source /insomnia001/depts/edu/users/yc4953/projects/allm-dgsc/scripts/activate_allm.sh
destination=outputs/sc-test256-a6000-v1
mkdir -p "$destination"
nvidia-smi > "$destination/gpu-${SLURM_JOB_ID}.txt"
nvidia-smi topo -m > "$destination/topology-${SLURM_JOB_ID}.txt"
python -m pip freeze > "$destination/packages-${SLURM_JOB_ID}.txt"
# Keep the existing single-H100 job held, never running the benchmark twice.
# If this allocation cannot load/generate correctly, release that fallback.
trap 'status=$?; if [ "$status" -ne 0 ]; then scontrol release 13934635 || true; fi' EXIT
python -u -m exp_code.baselines.batched_run \
  --data-root /insomnia001/depts/edu/users/yc4953/datasets \
  --checkpoint /insomnia001/depts/edu/users/yc4953/models/diffusiongemma-26B-A4B-it \
  --output "$destination" --batch-size 4 --model-shards 2
python -m exp_code.baselines.plot_benchmark_entropy --output "$destination"
scancel 13934635 || true
