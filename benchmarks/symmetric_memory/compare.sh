#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Run the four symm-mem implementations one at a time and print each run's
# table: torch CUDASymmetricMemory, torchpyimpl CUDA-IPC, torchpyimpl NCCL,
# torch NCCLSymmetricMemory.
#
# Each configuration gets its own `bench.py` process. That is the point rather
# than a convenience: torch's NCCLSymmetricMemory and the torchpyimpl NCCL
# backend register windows on the same communicator, so measuring them together
# inflates both. A process per run also means teardown is the kernel's job — driving several mp.spawn rounds
# from one interpreter leaves the process alive after the last one.
#
# Usage:
#   ./benchmarks/symmetric_memory/compare.sh                        # 8 ranks (or all GPUs), cached only
#   ./benchmarks/symmetric_memory/compare.sh --with-alloc-sweeps    # + alloc and cold rendezvous
#   RANKS="2 4 8" ./benchmarks/symmetric_memory/compare.sh          # sweep world size too
#
# Any extra arguments are passed through to every bench.py run.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$(dirname "$here")")"
py="$root/.venv/bin/python"
[ -x "$py" ] || { echo "no venv at $py; run uv sync first" >&2; exit 1; }

# backend:measure, in run order.
configs=("CUDA:torch" "CUDA:ours" "NCCL:ours" "NCCL:torch")
read -ra ranks <<< "${RANKS:-0}"   # 0 = let bench.py pick (all visible GPUs)

# One directory per invocation, so a session's runs stay together and nothing
# is ever overwritten. Old sessions accumulate; delete them yourself if you care.
session="${TORCHPYIMPL_BENCH_DIR:-/tmp/torchpyimpl-bench/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$session"
export TORCHPYIMPL_BENCH_DIR="$session"
echo "session: $session" >&2

total=$(( ${#configs[@]} * ${#ranks[@]} ))
n=0

for ws in "${ranks[@]}"; do
  for cfg in "${configs[@]}"; do
    backend="${cfg%%:*}"
    measure="${cfg##*:}"
    n=$((n + 1))
    ws_args=()
    [ "$ws" != "0" ] && ws_args=(--world-size "$ws")
    echo "[$n/$total] $measure $backend ${ws_args[*]:-(all GPUs)}" >&2

    "$py" "$here/bench.py" --backend "$backend" --measure "$measure" \
      "${ws_args[@]}" "$@"
  done
done

echo "raw per-rank results: $session" >&2
