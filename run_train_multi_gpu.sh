#!/usr/bin/env bash
set -euo pipefail

# Run from the project root, including when invoked from another directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# One training process per GPU; NUM_ENVS is the environment count per GPU.
# Select GPUs with CUDA_VISIBLE_DEVICES and match NPROC_PER_NODE to their count.
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NUM_ENVS="${NUM_ENVS:-2048}"

exec python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/rsl_rl/train.py \
  --task AME-G1-29DOF-v0 \
  --max_iterations 15000 \
  --distributed \
  --headless \
  --num_envs "${NUM_ENVS}" \
  "$@"
