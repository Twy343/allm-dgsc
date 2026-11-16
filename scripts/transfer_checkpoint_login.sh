#!/bin/bash
# Network transfer only; heavy verification remains in the existing Slurm job.
set -uo pipefail
export SLURM_CONF=/etc/slurm/slurm.conf
root=/insomnia001/depts/edu/users/yc4953
cd "$root/projects/allm-dgsc" || exit 1
status=0
nice -n 10 python3 -u -m exp_code.baselines.download_checkpoint \
  --output "$root/models/diffusiongemma-26B-A4B-it" --transfer-only || status=$?
printf 'TRANSFER_EXIT_CODE=%s\n' "$status"
# If a transfer fails, the existing resumable job can finish it safely.
scontrol release 13933051
exit "$status"
