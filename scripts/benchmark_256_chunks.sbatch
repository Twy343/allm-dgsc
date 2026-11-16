#!/bin/bash
#SBATCH --job-name=allm-sc-test256-chunk
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --constraint=R6000
#SBATCH --gres=gpu:R6000:1
#SBATCH --gres-flags=disable-binding
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-sc-test256-chunk-%j.log
set -euo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
source /insomnia001/depts/edu/users/yc4953/projects/allm-dgsc/scripts/activate_allm.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export JAX_COMPILATION_CACHE_DIR=/insomnia001/depts/edu/users/yc4953/hf_cache/jax_sc_test256
umask 077
destination=outputs/sc-test256-largegpu-v1
mkdir -p "$destination"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$destination/gpu-${SLURM_JOB_ID}.txt"
python -c 'import subprocess; sizes=subprocess.check_output(["nvidia-smi","--query-gpu=memory.total","--format=csv,noheader,nounits"],text=True).splitlines(); assert sizes and min(float(x) for x in sizes)>=75000, sizes'
python -m pip freeze > "$destination/packages-${SLURM_JOB_ID}.txt"
python -u -m exp_code.baselines.batched_run \
  --data-root /insomnia001/depts/edu/users/yc4953/datasets \
  --checkpoint /insomnia001/depts/edu/users/yc4953/models/diffusiongemma-26B-A4B-it \
  --output "$destination" --batch-size 4 --model-shards 1 --max-seconds 1200
complete=$(python -c 'import json; s=json.load(open("outputs/sc-test256-largegpu-v1/summary.json")); print(int(bool(s) and all(v["complete"] for v in s.values())))')
if [ "$complete" = 1 ]; then
  python -m exp_code.baselines.plot_benchmark_entropy --output "$destination"
  echo 'ALL_CONDITIONS_FINISHED'
else
  next=$(( ${ALLM_CHUNK_INDEX:-0} + 1 ))
  if [ "$next" -ge 8 ]; then echo 'CHUNK_BUDGET_EXHAUSTED_PARTIAL_RESULTS_SAVED'; exit 2; fi
  # Resume on the SAME GPU model; never merge different hardware into one run.
  feature=$(python -c 'import json; k=json.load(open("outputs/sc-test256-largegpu-v1/run.json"))["device_kind"][0]; assert "RTX PRO 6000" in k or "H100" in k,k; print("R6000" if "RTX PRO 6000" in k else "h100")')
  sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
    --constraint="$feature" --gres="gpu:$feature:1" --export="ALL,ALLM_CHUNK_INDEX=$next" \
    scripts/benchmark_256_chunks.sbatch > "$destination/next-job-${SLURM_JOB_ID}.txt"
  echo 'PARTIAL_SAVED_NEXT_JOB' "$(cat "$destination/next-job-${SLURM_JOB_ID}.txt")"
fi
