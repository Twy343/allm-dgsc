# Source inside a Slurm allocation; use the existing environment directly.
# This avoids depending on the retired anaconda/2023.09 modulefile.
allm_root=/insomnia001/depts/edu/users/yc4953
export CONDA_PREFIX="$allm_root/envs/allm"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export HF_HOME="$allm_root/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export LD_LIBRARY_PATH="$(find "$CONDA_PREFIX/lib/python3.12/site-packages/nvidia" -type d -name lib -printf '%p:' 2>/dev/null)${LD_LIBRARY_PATH:-}"
cd "$allm_root/projects/allm-dgsc"
