#!/bin/bash
# Execute inside a short, single-GPU interactive Slurm allocation.
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
source /insomnia001/depts/edu/users/yc4953/projects/allm-dgsc/scripts/activate_allm.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export JAX_COMPILATION_CACHE_DIR=/insomnia001/depts/edu/users/yc4953/hf_cache/jax_sc_test256
trap 'scontrol release 13937837 || true' EXIT
card=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$card" in
  *RTX*PRO*6000*) feature=R6000 ;;
  *H100*) feature=h100 ;;
  *) echo "Unexpected hardware: $card"; exit 1 ;;
esac
scontrol update JobId=13937837 Features="$feature" TresPerNode="gres/gpu:$feature:1"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# First batch warms the instrumented path; following batches give real latency.
# These are the first 12 FORMAL cases in the same fixed 256-case run.
python -u -m exp_code.baselines.batched_run \
  --data-root /insomnia001/depts/edu/users/yc4953/datasets \
  --checkpoint /insomnia001/depts/edu/users/yc4953/models/diffusiongemma-26B-A4B-it \
  --output outputs/sc-test256-largegpu-v1 --batch-size 4 --model-shards 1 --max-batches 3
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
