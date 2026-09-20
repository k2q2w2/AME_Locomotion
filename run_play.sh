#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
play_python="${PYTHON:-python}"
play_prefix="$("$play_python" -c 'import sys; print(sys.prefix)')"
# Isaac Sim may otherwise load the system C++ runtime before Conda's ICU/SQLite.
# Use the runtime belonging to this Python environment and preserve other preloads.
if [[ -d "$play_prefix/conda-meta" && -f "$play_prefix/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$play_prefix/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

if [[ $# -eq 0 ]]; then
    set -- --task AME-G1-29DOF-Play-v0 \
        --checkpoint pretrained/ame1.pt --num_envs 1 \
        --video --video_length 300 --save_attention_weights --vis_attention
fi

exec "$play_python" scripts/rsl_rl/play.py "$@"
