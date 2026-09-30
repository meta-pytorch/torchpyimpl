# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the multimem (multicast) kernels and barrier routing."""

from __future__ import annotations

import json
import os
import re
import socket
from contextlib import closing
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import pathlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torchpyimpl.distributed.symmetric_memory as symm
from torchpyimpl.distributed.symmetric_memory import _multimem, _signal


_SYM = re.compile(r"kernel_cutlass__[A-Za-z0-9_]+")
_N_ELEMS = 64
_ROUNDS = 5
_CHANNELS = 4
_SENTINEL = -1.0
# ~0.3 s on an H100: long enough that a barrier which does not wait reads the
# sentinel, short enough not to dominate the test.
_DELAY_CYCLES = 500_000_000


def test_fill_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="numel"):
        _multimem.fill(0x1000, 3, 1.0)
    with pytest.raises(ValueError, match="null multicast pointer"):
        _multimem.fill(0, 4, 1.0)
    with pytest.raises(ValueError, match="aligned"):
        _multimem.fill(0x1004, 4, 1.0)
    with pytest.raises(ValueError, match="at most"):
        _multimem.fill(0x1000, 2**33, 1.0)


def test_barrier_rejects_null_state() -> None:
    with pytest.raises(ValueError, match="null multicast barrier state"):
        _multimem.barrier(0, 0x1000, 2, 0)


def _kernel_symbols(compiled: Any) -> list[str]:
    return sorted(set(_SYM.findall(str(compiled.ir_module))))


def _only_symbol(compiled: Any) -> str:
    symbols = _kernel_symbols(compiled)
    assert len(symbols) == 1, f"expected exactly one kernel, got {symbols}"
    return symbols[0]


@pytest.mark.gpu
@pytest.mark.sm90
def test_multimem_kernels_are_named() -> None:
    torch.cuda.set_device(0)
    barrier = _only_symbol(_multimem._compiled_barrier(2, True, 0))
    fill = _only_symbol(_multimem._compiled_fill(32, 0))
    assert "torchpyimpl_multimem_barrier_kernel" in barrier, barrier
    assert "torchpyimpl_multimem_fill_kernel" in fill, fill
    # The CAS and multimem barriers must be distinguishable in a trace.
    cas = _only_symbol(_signal._compiled_barrier(2, 0, 32, True, 0))
    assert barrier != cas, barrier


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ordering_holds(
    hdl: symm.SymmMemHandle, rank: int, world_size: int, stamp: float, pg: Any
) -> bool:
    """Stamp our own buffer, barrier, then read every peer's. Rank 0 sleeps on
    the GPU first, so a barrier that does not wait reads its sentinel."""
    # Peers may still be reading this buffer from a previous step.
    dist.barrier(group=pg)
    mine = symm.get_buffer(hdl, rank, (_N_ELEMS,), torch.float32)
    mine.fill_(_SENTINEL)
    torch.cuda.synchronize()
    dist.barrier(group=pg)

    if rank == 0:
        torch.cuda._sleep(_DELAY_CYCLES)
    mine.fill_(stamp + rank)
    hdl.barrier()
    seen = [
        float(symm.get_buffer(hdl, p, (1,), torch.float32)[0].item())
        for p in range(world_size)
    ]
    return seen == [stamp + p for p in range(world_size)]


def _worker(rank: int, world_size: int, master_port: int, result_path: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device
    )
    try:
        pg = dist.new_group(ranks=list(range(world_size)))
        buf = symm.empty(_N_ELEMS, dtype=torch.float32, device=device)
        hdl = symm.rendezvous(buf, pg)
        has_mc = hdl.multicast_signal_pad_ptr != 0 and symm.has_multicast_support(hdl)
        flags: list[Any] = [None] * world_size
        dist.all_gather_object(flags, has_mc, group=pg)
        report: dict[str, Any] = {"rank": rank, "has_multicast": all(flags)}
        if not all(flags):
            symm.free_pool(device)
            _gather(report, pg, world_size, result_path)
            return

        calls: dict[str, int] = {"multimem": 0, "cas": 0}
        real_mm, real_cas = _multimem.barrier, _signal.barrier

        def count_mm(*a: Any, **k: Any) -> None:
            calls["multimem"] += 1
            real_mm(*a, **k)

        def count_cas(*a: Any, **k: Any) -> None:
            calls["cas"] += 1
            real_cas(*a, **k)

        _multimem.barrier, _signal.barrier = count_mm, count_cas
        try:
            dist.barrier(group=pg)
            report["mc_ordering"] = _ordering_holds(hdl, rank, world_size, 100.0, pg)
            report["routed_multimem"] = calls["multimem"]

            # The kernel subtracts world_size rather than zeroing, so repeated
            # rounds on one channel prove the counter was left reusable.
            for _ in range(_ROUNDS):
                hdl.barrier()
            torch.cuda.synchronize(device)

            # channel is a runtime argument: all channels share one compile.
            _multimem._compiled_barrier.cache_clear()
            for ch in range(_CHANNELS):
                hdl.barrier(channel=ch)
            torch.cuda.synchronize(device)
            report["compiles"] = _multimem._compiled_barrier.cache_info().currsize

            # A pending signal must not count as a barrier arrival. Rank 0
            # signals rank 1 and enters the barrier; rank 1 consumes the signal
            # only after rank 0's arrival has landed, then enters the barrier.
            dist.barrier(group=pg)
            if rank == 1:
                torch.cuda._sleep(_DELAY_CYCLES)
                hdl.wait_signal(0, timeout_ms=10_000)
            elif rank == 0:
                hdl.put_signal(1, timeout_ms=10_000)
            hdl.barrier(timeout_ms=10_000)
            torch.cuda.synchronize(device)
            report["signal_then_barrier_ok"] = True

            # Force the CAS fallback on the same (cached) handle.
            mc_pad = hdl.multicast_signal_pad_ptr
            hdl.multicast_signal_pad_ptr = 0
            before_cas = calls["cas"]
            dist.barrier(group=pg)
            report["cas_ordering"] = _ordering_holds(hdl, rank, world_size, 200.0, pg)
            report["routed_cas"] = calls["cas"] - before_cas
            hdl.multicast_signal_pad_ptr = mc_pad
        finally:
            _multimem.barrier, _signal.barrier = real_mm, real_cas

        # Multimem broadcast: rank 0 stores through the multicast pointer and
        # every rank must read the value from its own physical buffer.
        dist.barrier(group=pg)  # peers may still be reading the CAS stamps
        mine = symm.get_buffer(hdl, rank, (_N_ELEMS,), torch.float32)
        mine.zero_()
        torch.cuda.synchronize(device)
        dist.barrier(group=pg)
        if rank == 0:
            _multimem.fill(hdl.multicast_ptr, _N_ELEMS, 7.0)
            torch.cuda.synchronize(device)
        dist.barrier(group=pg)
        report["fill_ok"] = bool(torch.all(mine == 7.0).item())

        dist.barrier(group=pg)
        symm.free_pool(device)
        _gather(report, pg, world_size, result_path)
    finally:
        dist.destroy_process_group()


def _gather(report: dict[str, Any], pg: Any, world_size: int, result_path: str) -> None:
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, report, group=pg)
    if report["rank"] == 0:
        with open(result_path, "w") as f:
            json.dump({"reports": gathered}, f)


@pytest.mark.multi_gpu
@pytest.mark.peer_access
def test_multimem_barrier_routing_and_fill(tmp_path: pathlib.Path) -> None:
    world_size = min(torch.cuda.device_count(), 8)
    result_path = str(tmp_path / "reports.json")
    mp.spawn(
        _worker,
        args=(world_size, _free_port(), result_path),
        nprocs=world_size,
        join=True,
    )

    with open(result_path) as f:
        reports = json.load(f)["reports"]

    assert len(reports) == world_size
    if not reports[0]["has_multicast"]:
        pytest.skip("allocation has no multicast signal pad on this hardware")

    for r in reports:
        rank = r["rank"]
        assert r["routed_multimem"] == 1, (
            f"rank {rank}: barrier took the CAS path with a multicast pad"
        )
        assert r["mc_ordering"], f"rank {rank}: stamps not visible after multicast"
        assert r["routed_cas"] == 1, (
            f"rank {rank}: barrier did not fall back to CAS without a multicast pad"
        )
        assert r["cas_ordering"], f"rank {rank}: stamps not visible after CAS"
        assert r["compiles"] == 1, (
            f"rank {rank}: {r['compiles']} barrier compiles for {_CHANNELS} channels"
        )
        assert r["fill_ok"], f"rank {rank}: multimem broadcast did not land"
        assert r["signal_then_barrier_ok"], rank
