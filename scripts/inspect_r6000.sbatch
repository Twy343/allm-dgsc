#!/bin/bash
#SBATCH --job-name=allm-r6000-inventory
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --qos=burst
#SBATCH --constraint=R6000
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=/insomnia001/depts/edu/users/yc4953/logs/allm-r6000-inventory-%j.log
hostname
cat /proc/driver/nvidia/gpus/*/information
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
