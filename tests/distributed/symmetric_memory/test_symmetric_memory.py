# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone allocator and rendezvous tests on the CUDA-IPC backend."""

from __future__ import annotations

import json
import os
import socket
from contextlib import closing
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import pathlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cuda.bindings import driver

import torchpyimpl.distributed.symmetric_memory._symm_mem as sm
from torchpyimpl.distributed.symmetric_memory._comm import get_group_info
from torchpyimpl.distributed.symmetric_memory._cuda import (
    cu_mem_get_address_range,
    cu_memset_d8,
)
from torchpyimpl.distributed.symmetric_memory.backends import _cuda_ipc


_N_ELEMS = 1024
_NUM_TENSORS = 3
_NBYTES = 4001  # non-16-aligned user size -> round_up(., 16) = 4016


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_dist(rank: int, world_size: int, port: int) -> torch.device:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device
    )
    return device


def _gather_reports(report: dict[str, Any], pg: dist.ProcessGroup, out: str) -> None:
    gathered: list[Any] = [None] * dist.get_world_size(pg)
    dist.all_gather_object(gathered, report, group=pg)
    if dist.get_rank(pg) == 0:
        with open(out, "w") as f:
            json.dump({"reports": gathered}, f)


def _spawn(fn: Any, world_size: int, out: str, *args: Any) -> list[dict[str, Any]]:
    mp.spawn(fn, args=(world_size, _free_port(), out, *args), nprocs=world_size)
    with open(out) as f:
        return json.load(f)["reports"]


# ── standalone allocator ──────────────────────────────────────────────────────


def _alloc_worker(rank: int, result_path: str, transport: str) -> None:
    os.environ["TORCHPYIMPL_CUDA_IPC_TRANSPORT"] = transport
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=device)  # ensure a live primary context

    buf = sm.alloc(_NBYTES, device)
    off = sm.buffer_offset()
    base = cu_mem_get_address_range(buf)
    rec0, o0 = sm.find_covering(buf)
    reck, ok = sm.find_covering(buf + 64)
    pad_rec, _ = sm.find_covering(buf - 8)
    tail_rec, _ = sm.find_covering(buf + _NBYTES + 8)
    report: dict[str, Any] = {
        "base_nonzero": base != 0,
        "data_at_offset": (buf - base) == off,
        "exact_hit": rec0 is not None and o0 == 0,
        "interior_hit": (reck is rec0) and ok == 64,
        "pad_not_covered": pad_rec is None,
        "tail_not_covered": tail_rec is None,
    }
    sm.free(buf)
    recf, _ = sm.find_covering(buf)
    report["gone_after_free"] = recf is None
    report["records_after_free"] = sm._cache_stats()["records"]
    with open(result_path, "w") as f:
        json.dump(report, f)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "transport", [pytest.param("fabric", marks=pytest.mark.fabric), "fd"]
)
def test_alloc_find_covering_free(transport: str, tmp_path: pathlib.Path) -> None:
    result_path = str(tmp_path / "report.json")
    mp.spawn(_alloc_worker, args=(result_path, transport), nprocs=1)

    with open(result_path) as f:
        r = json.load(f)
    assert r["base_nonzero"], "cuMemGetAddressRange failed on the allocated buffer"
    assert r["data_at_offset"], "data ptr not at base + buffer_offset"
    assert r["exact_hit"], "exact find_covering miss"
    assert r["interior_hit"], "interior find_covering miss"
    assert r["pad_not_covered"], "ptr in the signal pad was reported as covered"
    assert r["tail_not_covered"], "ptr past buffer_size was reported as covered"
    assert r["gone_after_free"], "record still present after free"
    assert r["records_after_free"] == 0


def _standalone_rendezvous_worker(
    rank: int, world_size: int, port: int, out: str, transport: str
) -> None:
    # Must be set before the first allocation: cuMemCreate bakes the handle type
    # into the block.
    os.environ["TORCHPYIMPL_CUDA_IPC_TRANSPORT"] = transport
    _cuda_ipc.reset_transport()
    device = _init_dist(rank, world_size, port)
    try:
        pg = dist.new_group(ranks=list(range(world_size)))
        comm_ptr = get_group_info(pg, device, require_comm=False).comm_ptr
        nbytes = _N_ELEMS * 4
        buf = sm.alloc(nbytes, device)
        torch.cuda.synchronize(device)

        h1 = sm.rendezvous(buf, nbytes, comm_ptr, rank, world_size, device, group=pg)
        h2 = sm.rendezvous(buf, nbytes, comm_ptr, rank, world_size, device, group=pg)
        stats = sm._cache_stats()

        # Each rank fills its buffer with rank + 1, then reads every peer's.
        cu_memset_d8(buf, rank + 1, nbytes)
        dist.barrier(group=pg)
        peer_data_ok = True
        for peer in range(world_size):
            dst = torch.empty(nbytes, dtype=torch.uint8, device=device)
            (err,) = driver.cuMemcpy(dst.data_ptr(), h1.buffer_ptrs[peer], nbytes)
            ok = err == driver.CUresult.CUDA_SUCCESS
            peer_data_ok = peer_data_ok and ok and bool((dst == peer + 1).all())
        report: dict[str, Any] = {
            "rank": rank,
            "buffer_ptrs_len": len(h1.buffer_ptrs),
            "all_nonzero": all(p != 0 for p in h1.buffer_ptrs),
            "self_ptr": h1.buffer_ptrs[rank] == buf,
            "signal_pad_first": h1.buffer_ptr - h1.signal_pad_ptr == sm.buffer_offset(),
            "cache_same": h2 is h1,
            "cached_lookup": sm.get_cached_handle(buf, comm_ptr) is h1,
            "peer_alloc_infos": stats["peer_alloc_infos"],
            "transport": _cuda_ipc.get_transport(rank).value,
            "peer_data_ok": peer_data_ok,
        }
        dist.barrier(group=pg)
        sm.free(buf)
        after = sm._cache_stats()
        report["peer_alloc_infos_after_free"] = after["peer_alloc_infos"]
        report["records_after_free"] = after["records"]
        _gather_reports(report, pg, out)
        dist.destroy_process_group(pg)
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.peer_access
@pytest.mark.parametrize(
    "transport", [pytest.param("fabric", marks=pytest.mark.fabric), "fd"]
)
def test_standalone_rendezvous_cache_and_free(
    transport: str, tmp_path: pathlib.Path
) -> None:
    if transport == "fd" and not _cuda_ipc.cu_device_supports_posix_fd(0):
        pytest.skip("no POSIX-fd handle support on this device")
    world_size = min(torch.cuda.device_count(), 8)
    reports = _spawn(
        _standalone_rendezvous_worker,
        world_size,
        str(tmp_path / "reports.json"),
        transport,
    )
    assert len(reports) == world_size
    for r in reports:
        rank = r["rank"]
        assert r["buffer_ptrs_len"] == world_size, rank
        assert r["all_nonzero"], f"rank {rank}: a peer buffer pointer is 0"
        assert r["self_ptr"], f"rank {rank}: own buffer pointer is not the alloc"
        assert r["signal_pad_first"], rank
        assert r["cache_same"], f"rank {rank}: 2nd rendezvous returned a new handle"
        assert r["cached_lookup"], f"rank {rank}: get_cached_handle miss"
        assert r["peer_alloc_infos"] == 1, rank
        assert r["transport"] == transport, rank
        assert r["peer_data_ok"], f"rank {rank}: wrong data read from a peer"
        assert r["peer_alloc_infos_after_free"] == 0, rank
        assert r["records_after_free"] == 0, rank
