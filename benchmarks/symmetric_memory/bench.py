# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark: one symm-mem configuration, one run.

Times a single (backend, implementation) pair and prints one table. For the
four-way comparison — torch CUDASymmetricMemory, torchpyimpl CUDA-IPC, torchpyimpl
NCCL, torch NCCLSymmetricMemory — use ``compare.sh``, which invokes this script
once per configuration. Separate processes are deliberate: see that script.

``--backend {NCCL,CUDA}`` picks the symmetric-memory backend, ``--measure``
picks what is timed in it:

    ours    torchpyimpl (default)
    torch   torch's native symm-mem for the matching backend
    both    both in one process, printing a side-by-side ratio

``both`` is the only mode where the two share a process. The ratio is valid, but
on the NCCL backend torch and torchpyimpl register windows on the same borrowed
communicator and contend, so the absolutes inflate — use separate runs when the
absolute numbers matter.

Three sweeps:

* **rendezvous (cached)** — always run. A cache-hit rendezvous: no collective, no
  device work, so this measures interpreter cost. This is the production path.
* **alloc** (``--with-alloc-sweeps``) — one allocate per iteration:
  ``ncclMemAlloc`` on NCCL, ``cuMemCreate`` on CUDA-IPC, ``empty`` for torch.
  Every tensor is held, so each iteration is a pool miss. The frees run after
  the timed loop.
* **rendezvous (cold)** (``--with-alloc-sweeps``) — allocate + rendezvous each
  iteration, so the cross-rank step fires every time; the frees run after the
  timed loop. **The alloc is inside this measurement** — subtract the alloc row
  to isolate the rendezvous.

The alloc sweeps are off by default: they churn the symmetric allocator on a live
comm and can trigger a non-deterministic NCCL illegal access on some builds. They
also need every rank to finish allocating before any rank frees, or NCCL's
symmetric allocator faults; a barrier after the timed loop does this, outside
the reported time.

Timing:

* Wall clock via ``time.perf_counter_ns`` — CUDA events cannot time Python and
  pybind work, which is most of what is under test.
* **p20/p50/p80**, not mean +/- stdev: host-side distributions are heavy-tailed.
* Cached is **auto-sized** — estimate one iteration, then derive warmup and
  repeat counts from a 25 ms / 100 ms budget. Alloc and cold use a **fixed** 3
  warmup / 10 timed; at ~100 ms an iteration a time budget is unusable.
* ``host_only`` on the cached sweep skips the per-iteration ``cuda.synchronize``
  (~1 us, comparable to the value being measured, and there is no device work to
  synchronize with).
* A ``dist.barrier`` precedes every alloc and cold iteration, keeping one
  iteration's teardown skew out of the next.

Output: raw per-rank JSON under ``$TORCHPYIMPL_BENCH_DIR`` (default
``/tmp/torchpyimpl-bench``), named for backend, what was measured, sweeps, world
size and timestamp, with the path printed as the last line. The console table
shows rank 0 only; the JSON carries every rank.

Usage::

    ./benchmarks/symmetric_memory/compare.sh --with-alloc-sweeps   # 4-way comparison
    python benchmarks/symmetric_memory/bench.py          # torchpyimpl CUDA, cached
    python benchmarks/symmetric_memory/bench.py --backend CUDA --measure torch
    python benchmarks/symmetric_memory/bench.py --backend CUDA --with-alloc-sweeps --max-log2 33

Buffer sizes sweep powers of two (``--min-log2`` .. ``--max-log2``, default 4 KiB
.. 1 GiB); alloc and cold also print a ``GB/s`` setup-throughput column.
``--world-size`` sets the rank count (default: all visible GPUs, capped at 8);
``compare.sh`` drives a sweep of them via ``RANKS``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import tempfile
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as torch_symm_mem
import torch.multiprocessing as mp

import torchpyimpl.distributed.symmetric_memory as symm


# ── Knobs ─────────────────────────────────────────────────────────────────────
# Buffer sizes are swept as powers of two (``--min-log2`` .. ``--max-log2``); the
# default spans 4 KiB .. 1 GiB, a wide range for a roofline. Ceiling: each rank's
# *physical* footprint is ~one buffer — the peer P2P mappings and the multicast
# object are virtual-address only (windows into peers' HBM), so VA is
# ``(world_size + 1) * N`` but physical is just ``N`` per rank. So N can approach
# free HBM per GPU (~183 GiB on GB200, i.e. up to ~2^37); in practice the alloc /
# cold sweeps get slow well before that, so raise ``--max-log2`` deliberately.
# Raw per-rank results land here. The printed table only shows rank 0's
# quantiles; the JSON carries every rank, so a straggler stays visible.
# Results accumulate; nothing here is ever overwritten or removed.
# TORCHPYIMPL_BENCH_DIR groups one session's runs (compare.sh sets it).
_RESULT_DIR: str = os.environ.get(
    "TORCHPYIMPL_BENCH_DIR", os.path.join(tempfile.gettempdir(), "torchpyimpl-bench")
)

# Which implementation a run times. Keeping them in separate processes is the
# point of BOTH vs OURS/TORCH: torch's NCCLSymmetricMemory registers windows on
# the same borrowed communicator torchpyimpl uses, and measured side by side the
# two contend — torchpyimpl's NCCL cold rendezvous inflates ~1.5x and its alloc
# ~6x. The CUDA backends share no such state and are unaffected.
OURS: str = "ours"
TORCH: str = "torch"
BOTH: str = "both"

_DEFAULT_MIN_LOG2 = 12  # 4 KiB
_DEFAULT_MAX_LOG2 = 30  # 1 GiB

# Cold rendezvous and alloc fire a real cuMem/collective op per iter (ncclMemAlloc
# / ncclCommWindowRegister). These are expensive AND stress NCCL's symmetric
# allocator, so keep the counts modest — a handful of samples already gives a
# stable p50 for ops in the tens-of-us-to-ms range.
ALLOC_N_WARMUP = 3
ALLOC_N_REPEAT = 10
COLD_N_WARMUP = 3
COLD_N_REPEAT = 10

# Cached rendezvous is host-only for both backends; auto-size to a time budget.
CACHED_WARMUP_MS = 25
CACHED_REP_MS = 100


# ── Timing helpers ────────────────────────────────────────────────────────────


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _quantiles(times_ms: list[float], qs=(0.2, 0.5, 0.8)) -> tuple[float, ...]:
    """Linear-interpolated quantiles over a sample list."""
    sorted_t = sorted(times_ms)
    n = len(sorted_t)

    def at(q: float) -> float:
        idx = q * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return sorted_t[lo] * (1 - frac) + sorted_t[hi] * frac

    return tuple(at(q) for q in qs)


def _do_bench_walltime(
    fn: Callable[[], None],
    device: torch.device,
    warmup_ms: int = CACHED_WARMUP_MS,
    rep_ms: int = CACHED_REP_MS,
    host_only: bool = False,
) -> tuple[float, float, float]:
    """Auto-sized wall-clock bench for host-bound work. Returns p20/p50/p80 ms.

    Uses ``time.perf_counter_ns`` (CUDA events cannot time pure Python/pybind
    work). ``host_only=True`` skips the per-iter ``cuda.synchronize`` calls (each
    is ~1 us, which would distort a ~10 us cached-rendezvous reading)."""
    for _ in range(2):
        fn()

    t0 = time.perf_counter_ns()
    for _ in range(5):
        fn()
    if not host_only:
        torch.cuda.synchronize(device)
    est_ms = max((time.perf_counter_ns() - t0) / 5 / 1e6, 1e-3)

    n_warmup = max(1, int(warmup_ms / est_ms))
    n_repeat = max(5, int(rep_ms / est_ms))

    for _ in range(n_warmup):
        fn()
    if not host_only:
        torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(n_repeat):
        if not host_only:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter_ns()
        fn()
        if not host_only:
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter_ns() - t0) / 1e6)

    return _quantiles(times_ms)


def _do_bench_fixed_iters(
    fn: Callable[[], None],
    device: torch.device,
    n_warmup: int,
    n_repeat: int,
    pre_iter_barrier: dist.ProcessGroup | None = None,
) -> tuple[float, float, float]:
    """Fixed-iteration wall-clock bench for slow/collective ops. Returns
    p20/p50/p80 ms. ``pre_iter_barrier`` issues ``dist.barrier`` before each iter
    (warmup and timed) so a collective's measurement is not polluted by the
    previous iter's max-rank teardown tail — and, for the alloc scenario, so
    ``ncclMemAlloc``/``ncclMemFree`` on the live comm run in lockstep across ranks
    (unsynchronized alloc/free corrupts NCCL's symmetric allocator)."""
    for _ in range(n_warmup):
        if pre_iter_barrier is not None:
            dist.barrier(group=pre_iter_barrier)
        fn()
        # Keep warmup iters from piling up async window ops that race across
        # ranks (the timed loop already synchronizes after each fn()).
        torch.cuda.synchronize(device)
    torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(n_repeat):
        if pre_iter_barrier is not None:
            dist.barrier(group=pre_iter_barrier)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter_ns() - t0) / 1e6)

    return _quantiles(times_ms)


# ── Scenarios ─────────────────────────────────────────────────────────────────
#
# Each backend allocates from its own symm-mem MemPool — a rendezvous requires
# the tensor to come from its own pool, so the two cannot share an allocation.
# The cold-rendezvous rows therefore include one alloc per iter for both;
# subtract the alloc-only row to isolate the rendezvous-wrapper delta.


def _bench_alloc_torch(nbytes: int, group, device: torch.device):
    """torch symm-mem allocation, freed outside the timed region.

    Only the allocate is timed. Freeing inside the loop would mean timing
    ``empty_cache``, which sweeps every cached segment on the device rather than
    releasing just this one — measured at ~75 ms of an 82 ms loop, i.e. the
    column would report cache teardown rather than allocation. torchpyimpl's free is
    targeted and has no equivalent cost, so timing the free makes the two columns
    incomparable."""
    keep: list = []

    def one() -> None:
        keep.append(torch_symm_mem.empty(nbytes, dtype=torch.uint8, device=device))

    try:
        return _do_bench_fixed_iters(
            one, device, ALLOC_N_WARMUP, ALLOC_N_REPEAT, pre_iter_barrier=group
        )
    finally:
        # Every rank must finish allocating before any rank frees, or NCCL's
        # symmetric allocator faults.
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        keep.clear()
        torch.cuda.empty_cache()


def _bench_alloc_ours(nbytes: int, group, device: torch.device):
    """torchpyimpl allocation through the symmetric ``MemPool``, i.e. the custom
    pluggable allocator — the same shape as torch, whose ``empty`` also allocates
    from an implicit ``MemPool`` (``TORCH_SYMMMEM_IMPLICIT_POOL``, on by default).
    Using the standalone allocator here instead would put a caching allocator in
    front of one side only.

    Every tensor is held for the duration, so no block can be recycled and each
    iteration is a pool miss — a real ``ncclMemAlloc`` / ``cuMemCreate``. Teardown
    is deferred out of the timed region (see :func:`_bench_alloc_torch`)."""
    keep: list = []

    def one() -> None:
        keep.append(symm.empty(nbytes, dtype=torch.uint8, device=device))

    try:
        return _do_bench_fixed_iters(
            one, device, ALLOC_N_WARMUP, ALLOC_N_REPEAT, pre_iter_barrier=group
        )
    finally:
        # Every rank must finish allocating before any rank frees, or NCCL's
        # symmetric allocator faults.
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        keep.clear()
        symm.free_pool(device)
        torch.cuda.empty_cache()


def _bench_rendezvous_cold_torch(nbytes: int, group, device: torch.device):
    """torch cold rendezvous: fresh ``empty`` + ``rendezvous`` per iteration, so
    the cache misses and the full collective fires every time.

    Teardown is deferred out of the timed region for the same reason as
    :func:`_bench_alloc_torch`: the free is ``empty_cache``, a device-wide sweep
    costing ~75 ms, which is not a rendezvous cost and has no torchpyimpl
    counterpart. Timing it made torch's cold column ~1.6x torchpyimpl's when the
    two rendezvous paths are in fact within a few percent."""
    group_name = group.group_name
    keep: list = []

    def one() -> None:
        buf = torch_symm_mem.empty(nbytes, dtype=torch.uint8, device=device)
        torch.cuda.synchronize(device)
        keep.append((buf, torch_symm_mem.rendezvous(buf, group=group_name)))
        torch.cuda.synchronize(device)

    try:
        return _do_bench_fixed_iters(
            one, device, COLD_N_WARMUP, COLD_N_REPEAT, pre_iter_barrier=group
        )
    finally:
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        keep.clear()
        torch.cuda.empty_cache()


def _bench_rendezvous_cold_ours(nbytes: int, group, device: torch.device):
    """torchpyimpl cold rendezvous through the public API — ``torchpyimpl.empty`` (the
    pluggable allocator behind a ``MemPool``) plus ``torchpyimpl.rendezvous``, so the
    call shape matches torch's ``empty`` + ``rendezvous``. Cache misses every
    iteration, so the cross-rank step fires each time. Teardown deferred."""
    keep: list = []

    def one() -> None:
        t = symm.empty(nbytes, dtype=torch.uint8, device=device)
        torch.cuda.synchronize(device)
        symm.rendezvous(t, group)
        torch.cuda.synchronize(device)
        keep.append(t)

    try:
        return _do_bench_fixed_iters(
            one, device, COLD_N_WARMUP, COLD_N_REPEAT, pre_iter_barrier=group
        )
    finally:
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
        keep.clear()
        symm.free_pool(device)
        torch.cuda.empty_cache()


def _bench_cached_torch(nbytes: int, group, device: torch.device):
    """torch C++ cache-hit rendezvous (host-side map lookup). Primes once, times
    the second-onward call."""
    group_name = group.group_name
    buf = torch_symm_mem.empty(nbytes, dtype=torch.uint8, device=device)
    _ = torch_symm_mem.rendezvous(buf, group=group_name)  # prime

    def one() -> None:
        torch_symm_mem.rendezvous(buf, group=group_name)

    try:
        return _do_bench_walltime(one, device, host_only=True)
    finally:
        torch.cuda.empty_cache()


def _bench_cached_ours(nbytes: int, group, device: torch.device):
    """torchpyimpl cache-hit rendezvous. The hot path is collective-free — a
    memoized group lookup and a handle-cache hit — so it auto-sizes per rank like
    torch's (no barrier needed)."""
    t = symm.empty(nbytes, dtype=torch.uint8, device=device)
    _ = symm.rendezvous(t, group)  # prime

    def one() -> None:
        symm.rendezvous(t, group)

    try:
        return _do_bench_walltime(one, device, host_only=True)
    finally:
        symm.free_pool(device)
        torch.cuda.empty_cache()


# ── Sweeps (one row builder per scenario; keeps the worker a short orchestrator)


def _row(nbytes: int, ours_q=None, torch_q=None) -> dict:
    row: dict = {"nbytes": nbytes}
    if ours_q is not None:
        op20, op50, op80 = ours_q
        row["ours_p20_ms"] = op20
        row["ours_p50_ms"] = op50
        row["ours_p80_ms"] = op80
    if torch_q is not None:
        tp20, tp50, tp80 = torch_q
        row["torch_p20_ms"] = tp20
        row["torch_p50_ms"] = tp50
        row["torch_p80_ms"] = tp80
    return row


def _run_alloc_sweep(group, device, measure: str, buffer_sizes) -> list[dict]:
    return [
        _row(
            n,
            _bench_alloc_ours(n, group, device) if measure != TORCH else None,
            _bench_alloc_torch(n, group, device) if measure != OURS else None,
        )
        for n in buffer_sizes
    ]


def _run_cold_sweep(group, device, measure: str, buffer_sizes) -> list[dict]:
    return [
        _row(
            n,
            _bench_rendezvous_cold_ours(n, group, device) if measure != TORCH else None,
            _bench_rendezvous_cold_torch(n, group, device) if measure != OURS else None,
        )
        for n in buffer_sizes
    ]


def _run_cached_sweep(group, device, measure: str, buffer_sizes) -> list[dict]:
    return [
        _row(
            n,
            _bench_cached_ours(n, group, device) if measure != TORCH else None,
            _bench_cached_torch(n, group, device) if measure != OURS else None,
        )
        for n in buffer_sizes
    ]


def _bench_worker(
    rank,
    world_size,
    master_port,
    result_path,
    measure: str,
    with_alloc_sweeps: bool,
    backend: str,
    buffer_sizes: list,
) -> None:
    # The lib logs alloc / window register / free at INFO on every iter, which
    # floods a sweep of hundreds of alloc+rendezvous cycles. Quiet it for the
    # benchmark — we only care about the timing table.
    logging.getLogger("torchpyimpl").setLevel(logging.WARNING)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # torch's native comparison uses torch's *matching* C++ backend, so torchpyimpl
    # NCCL is timed against torch NCCLSymmetricMemory and torchpyimpl CUDA against
    # torch CUDASymmetricMemory (apples-to-apples).
    use_torch = measure in (TORCH, BOTH)
    torch_native_note = None
    if use_torch:
        # torch's symm-mem defaults to the CUDA backend (CUDASymmetricMemory), and
        # set_backend takes only explicit overrides — set_backend("CUDA") is
        # rejected since CUDA is the implicit default, not a named allocator. So
        # only switch when torch's active backend differs from the one we want;
        # for CUDA we leave the default in place.
        try:
            if torch_symm_mem.get_backend(device) != backend:
                torch_symm_mem.set_backend(backend)
        except Exception as e:
            # A named backend torch's build didn't register (e.g. NCCL on a build
            # without it) — skip the comparison gracefully rather than crashing.
            use_torch = False
            measure = OURS
            torch_native_note = (
                f"torch-native comparison skipped: torch has no {backend} "
                f"symm-mem backend in this build ({type(e).__name__}: {e})"
            )
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device
    )
    try:
        # Eager-init the comm so neither backend pays lazy-init cost.
        group = dist.group.WORLD
        if group is None:
            raise RuntimeError("default process group was not initialized")
        symm.set_backend(backend)  # NCCL windows or CUDA-IPC VMM

        # The cached-rendezvous sweep is host-only and robust. The alloc + cold
        # sweeps churn the symmetric allocator (ncclMemAlloc / window register or
        # cuMemCreate / multicast bind) on the live comm and are opt-in
        # (--with-alloc-sweeps): on some torch/NCCL builds that churn triggers a
        # non-deterministic illegal memory access inside NCCL's symmetric
        # allocator (torchpyimpl's normal one-shot lifecycle is fine).
        results = {
            "rank": rank,
            "world_size": world_size,
            "backend": backend,
            "with_torch": use_torch,
            "measure": measure,
            "torch_native_note": torch_native_note,
            "free_hbm_gib": torch.cuda.mem_get_info(device)[0] / 2**30,
            "cached": _run_cached_sweep(group, device, measure, buffer_sizes),
        }
        if with_alloc_sweeps:
            results["alloc"] = _run_alloc_sweep(group, device, measure, buffer_sizes)
            results["rendezvous"] = _run_cold_sweep(
                group, device, measure, buffer_sizes
            )

        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, results, group=group)
        if rank == 0:
            with open(result_path, "w") as f:
                json.dump({"reports": gathered}, f)
    finally:
        dist.destroy_process_group()


# ── Output (rank 0) ───────────────────────────────────────────────────────────


def _fmt_ms(p20: float, p50: float, p80: float) -> str:
    """us when sub-ms, ms otherwise."""
    if p50 < 1.0:
        return f"{p50 * 1000:>7.2f} us (p20={p20 * 1000:.2f} p80={p80 * 1000:.2f})"
    return f"{p50:>7.2f} ms (p20={p20:.2f} p80={p80:.2f})"


def _fmt_bytes(n: int) -> str:
    """Human-readable power-of-two size, e.g. 4 KiB / 512 MiB / 1 GiB."""
    for div, unit in ((2**30, "GiB"), (2**20, "MiB"), (2**10, "KiB")):
        if n >= div:
            return f"{n / div:g} {unit}"
    return f"{n} B"


def _gbps(nbytes: int, p50_ms: float) -> str:
    """Setup throughput = bytes / p50 time. For alloc/cold this is the roofline
    metric (how fast a buffer of this size is stood up); a plateau marks the
    bandwidth-bound regime, a rising line the latency-bound one."""
    if p50_ms <= 0:
        return "-"
    return f"{nbytes / (p50_ms * 1e6):.1f} GB/s"


def _print_table(reports: list[dict]) -> None:
    bar = "=" * 120
    rank0 = reports[0]
    backend = rank0["backend"]
    measure = rank0.get("measure", BOTH if rank0["with_torch"] else OURS)
    use_torch = measure in (TORCH, BOTH)
    use_ours = measure in (OURS, BOTH)
    both = use_torch and use_ours
    ours_label = "torchpyimpl"
    impl_label = "nccl4py windows" if backend == "NCCL" else "CUDA-IPC VMM"
    print()
    print(bar)
    if use_torch:
        print(
            f"torchpyimpl vs torch symm-mem benchmark — {rank0['world_size']} ranks — "
            f"backend={backend}. Format: p50 (p20, p80)."
        )
    else:
        hint = (
            " (pass --measure torch|both for the torch C++ comparison)"
            if backend == "NCCL"
            else ""
        )
        print(
            f"torchpyimpl symm-mem benchmark — {rank0['world_size']} ranks — "
            f"backend={backend} ({impl_label}). Format: p50 (p20, p80).{hint}"
        )
    free_hbm = rank0.get("free_hbm_gib")
    if free_hbm:
        print(
            f"free HBM/GPU: {free_hbm:.0f} GiB — a rank's physical footprint is ~one "
            "buffer (peer + multicast maps are VA-only), so max N ~ free HBM "
            "(raise --max-log2 to push toward it)."
        )
    note = rank0.get("torch_native_note")
    if note:
        print(f"NOTE: {note}")
    print(bar)

    for title, key, show_tput in [
        ("Allocation only (no rendezvous)", "alloc", True),
        ("Rendezvous (cold; includes one alloc per iter)", "rendezvous", True),
        ("Rendezvous (cached; cache-hit path)", "cached", False),
    ]:
        if key not in rank0:
            continue
        print()
        print(title)
        print("-" * 120)
        header = [f"{'buffer_size':>12}"]
        if use_torch:
            header.append(f"{'torch C++':>32}")
        if use_ours:
            header.append(f"{ours_label:>32}")
        if show_tput:
            header.append(f"{'GB/s':>14}")
        if both:
            header.append(f"{'ours/torch':>15}")
        print("  ".join(header))
        for row in rank0[key]:
            cells = [f"{_fmt_bytes(row['nbytes']):>12}"]
            if use_torch:
                cells.append(
                    f"{_fmt_ms(row['torch_p20_ms'], row['torch_p50_ms'], row['torch_p80_ms']):>32}"
                )
            if use_ours:
                cells.append(
                    f"{_fmt_ms(row['ours_p20_ms'], row['ours_p50_ms'], row['ours_p80_ms']):>32}"
                )
            if show_tput:
                key_p50 = "ours_p50_ms" if use_ours else "torch_p50_ms"
                cells.append(f"{_gbps(row['nbytes'], row[key_p50]):>14}")
            if both:
                ratio = (
                    row["ours_p50_ms"] / row["torch_p50_ms"]
                    if row["torch_p50_ms"] > 0
                    else float("inf")
                )
                cells.append(f"{ratio:>14.2f}x")
            print("  ".join(cells))
    if "alloc" not in rank0:
        print()
        print(
            "(alloc + cold-rendezvous sweeps skipped — pass --with-alloc-sweeps "
            "to run them; they churn the symmetric allocator and can hit a "
            "non-deterministic illegal access on some builds)"
        )
    print()
    print(bar)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measure",
        choices=[OURS, TORCH, BOTH],
        default=OURS,
        help="Which implementation to time: torchpyimpl (default), torch's native "
        "symm-mem for the matching backend, or both in one process. BOTH gives a "
        "side-by-side ratio, but on the NCCL backend the two contend on the "
        "shared communicator and both absolutes inflate — use separate runs "
        "(see compare.sh) when the absolute numbers matter.",
    )
    parser.add_argument(
        "--with-alloc-sweeps",
        action="store_true",
        help="Also run the alloc + cold-rendezvous sweeps. These churn the "
        "symmetric allocator on the live comm and can trigger a "
        "non-deterministic NCCL illegal memory access on some torch builds; off "
        "by default (the cached-rendezvous sweep always runs).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Ranks to run with (default: all visible GPUs, capped at 8).",
    )
    parser.add_argument(
        "--backend",
        choices=["NCCL", "CUDA"],
        default="CUDA",
        help="Which torchpyimpl backend to benchmark: the CUDA-IPC VMM backend "
        "(default) or NCCL windows. --measure torch|both compares against "
        "torch's matching C++ backend.",
    )
    parser.add_argument(
        "--min-log2",
        type=int,
        default=_DEFAULT_MIN_LOG2,
        help=f"Smallest buffer size as a power of two (default {_DEFAULT_MIN_LOG2} "
        f"= {_fmt_bytes(1 << _DEFAULT_MIN_LOG2)}).",
    )
    parser.add_argument(
        "--max-log2",
        type=int,
        default=_DEFAULT_MAX_LOG2,
        help=f"Largest buffer size as a power of two (default {_DEFAULT_MAX_LOG2} "
        f"= {_fmt_bytes(1 << _DEFAULT_MAX_LOG2)}). The ceiling is free HBM per GPU "
        "(~2^37 on GB200); large sizes make the alloc/cold sweeps slow.",
    )
    return parser.parse_args(argv)


def _result_path(backend: str, measure: str, with_alloc: bool, world_size: int) -> str:
    """Where a run's raw JSON goes.

    Named for what produced it — backend, what was measured, sweeps, timestamp —
    because a bench session is several runs and an anonymous temp file cannot be
    matched back to its command afterwards."""
    parts = [
        "torchpyimpl-bench",
        backend,
        {OURS: "solo", TORCH: "torch", BOTH: "native"}[measure],
    ]
    if with_alloc:
        parts.append("alloc")
    parts.append(f"w{world_size}")
    parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    Path(_RESULT_DIR).mkdir(parents=True, exist_ok=True)
    return os.path.join(_RESULT_DIR, "-".join(parts) + ".json")


def _run_one(
    world_size: int,
    buffer_sizes: list[int],
    backend: str,
    measure: str,
    with_alloc: bool,
) -> tuple[list[dict], str]:
    """Spawn one isolated benchmark run and return (reports, json path)."""
    path = _result_path(backend, measure, with_alloc, world_size)
    mp.spawn(
        _bench_worker,
        args=(
            world_size,
            _free_port(),
            path,
            measure,
            with_alloc,
            backend,
            buffer_sizes,
        ),
        nprocs=world_size,
        join=True,
    )
    with open(path) as f:
        return json.load(f)["reports"], path


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available; benchmark requires GPUs.", file=sys.stderr)
        sys.exit(1)
    available = min(torch.cuda.device_count(), 8)
    world_size = args.world_size if args.world_size is not None else available
    if world_size < 2 or world_size > available:
        print(
            f"--world-size must be between 2 and {available}; got {world_size}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.min_log2 > args.max_log2:
        print("--min-log2 must be <= --max-log2", file=sys.stderr)
        sys.exit(1)
    buffer_sizes = [1 << e for e in range(args.min_log2, args.max_log2 + 1)]

    reports, path = _run_one(
        world_size, buffer_sizes, args.backend, args.measure, args.with_alloc_sweeps
    )
    _print_table(reports)
    print(f"raw per-rank results: {path}")


if __name__ == "__main__":
    main()
