# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-allocation access patterns and allocation lifecycle on every backend."""

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

import torchpyimpl.distributed.symmetric_memory as api
from torchpyimpl.distributed.symmetric_memory._symm_mem import _cache_stats


_BASE = 100_000


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(
    worker: Any, world_size: int, tmp_path: pathlib.Path, backend: str
) -> list[dict[str, Any]]:
    master_port = _free_port()
    result_path = str(tmp_path / "reports.json")
    mp.spawn(
        worker,
        args=(world_size, master_port, result_path, backend),
        nprocs=world_size,
        join=True,
    )
    with open(result_path) as f:
        return json.load(f)["reports"]


def _init(rank: int, world_size: int, master_port: int, backend: str) -> torch.device:
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
    api.set_backend(backend)
    return device


_N = 1024
_OFF = 128
_M = 64


def _worker_kernel_behaviors(
    rank: int, world_size: int, master_port: int, result_path: str, backend: str
) -> None:
    device = _init(rank, world_size, master_port, backend)
    try:
        pg = dist.new_group(ranks=list(range(world_size)))

        t1 = api.empty(_N, dtype=torch.float32, device=device)
        idx = torch.arange(_N, dtype=torch.float32, device=device)
        t1.copy_((rank + 1) * _BASE + idx)
        h1 = api.rendezvous(t1, pg)
        torch.cuda.synchronize(device)
        dist.barrier(group=pg)

        k = torch.arange(_M, dtype=torch.float32, device=device)
        interior_ok = all(
            bool(
                torch.equal(
                    api.get_buffer(h1, p, (_M,), torch.float32, storage_offset=_OFF),
                    (p + 1) * _BASE + _OFF + k,
                )
            )
            for p in range(world_size)
        )

        own_pad = api.get_signal_pad(h1, rank, (4,), torch.uint32)
        own_pad.fill_(rank + 1)
        torch.cuda.synchronize(device)
        dist.barrier(group=pg)
        peer = (rank + 1) % world_size
        peer_pad = api.get_signal_pad(h1, peer, (4,), torch.uint32)
        signal_pad_peer_ok = int(peer_pad[0].item()) == (peer + 1)

        report: dict[str, Any] = {
            "rank": rank,
            "interior_ok": interior_ok,
            "signal_pad_peer_ok": signal_pad_peer_ok,
            "has_multicast": api.has_multicast_support(h1),
        }
        dist.barrier(group=pg)
        del t1
        report["free_status"] = api.free_pool(device, pg).value
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, report, group=pg)
        if rank == 0:
            with open(result_path, "w") as f:
                json.dump({"reports": gathered}, f)
        dist.destroy_process_group(pg)
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.peer_access
@pytest.mark.parametrize("backend", ["CUDA", "NCCL"])
def test_kernel_behaviors_per_allocation(backend: str, tmp_path: pathlib.Path) -> None:
    world_size = min(torch.cuda.device_count(), 8)
    reports = _spawn(_worker_kernel_behaviors, world_size, tmp_path, backend)
    assert len(reports) == world_size
    for r in reports:
        rank = r["rank"]
        assert r["interior_ok"], f"rank {rank}: interior-offset peer view wrong"
        assert r["signal_pad_peer_ok"], f"rank {rank}: peer signal pad not visible"
        assert r["free_status"] == "freed", f"rank {rank}: {r['free_status']}"


_NUM_ALLOCS = 3
_LN = 512


def _worker_lifecycle(
    rank: int, world_size: int, master_port: int, result_path: str, backend: str
) -> None:
    device = _init(rank, world_size, master_port, backend)
    try:
        pg = dist.new_group(ranks=list(range(world_size)))

        def _peers_ok(hdl: Any) -> bool:
            return all(
                bool(torch.all(api.get_buffer(hdl, p, (_LN,), torch.float32) == p + 1))
                for p in range(world_size)
            )

        tensors = [
            api.empty(_LN, dtype=torch.float32, device=device)
            for _ in range(_NUM_ALLOCS)
        ]
        for t in tensors:
            t.fill_(float(rank + 1))
        torch.cuda.synchronize(device)
        handles = [api.rendezvous(t, pg) for t in tensors]
        dist.barrier(group=pg)
        infos_in_flight = _cache_stats()["peer_alloc_infos"]
        peers_ok = all(_peers_ok(h) for h in handles)

        dist.barrier(group=pg)
        del tensors, t
        free_status = api.free_pool(device, pg).value
        stats_after_free = _cache_stats()

        t_new = api.empty(_LN, dtype=torch.float32, device=device)
        t_new.fill_(float(rank + 1))
        torch.cuda.synchronize(device)
        h_new = api.rendezvous(t_new, pg)
        dist.barrier(group=pg)
        realloc_ok = _peers_ok(h_new)
        infos_after_realloc = _cache_stats()["peer_alloc_infos"]

        report: dict[str, Any] = {
            "rank": rank,
            "infos_in_flight": infos_in_flight,
            "peers_ok": peers_ok,
            "free_status": free_status,
            "infos_after_free": stats_after_free["peer_alloc_infos"],
            "records_after_free": stats_after_free["records"],
            "pending_after_free": stats_after_free["pending"],
            "realloc_ok": realloc_ok,
            "infos_after_realloc": infos_after_realloc,
        }
        dist.barrier(group=pg)
        del t_new
        report["final_free_status"] = api.free_pool(device, pg).value
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, report, group=pg)
        if rank == 0:
            with open(result_path, "w") as f:
                json.dump({"reports": gathered}, f)
        dist.destroy_process_group(pg)
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.peer_access
@pytest.mark.parametrize("backend", ["CUDA", "NCCL"])
def test_lifecycle_and_multi_resource(backend: str, tmp_path: pathlib.Path) -> None:
    world_size = min(torch.cuda.device_count(), 8)
    reports = _spawn(_worker_lifecycle, world_size, tmp_path, backend)
    assert len(reports) == world_size
    for r in reports:
        rank = r["rank"]
        assert r["infos_in_flight"] == _NUM_ALLOCS, (
            f"rank {rank}: {r['infos_in_flight']} peer sets for {_NUM_ALLOCS} allocs"
        )
        assert r["peers_ok"], f"rank {rank}: a peer's buffer has the wrong data"
        assert r["free_status"] == "freed", f"rank {rank}: {r['free_status']}"
        assert r["infos_after_free"] == 0, (
            f"rank {rank}: peer sets left after free_pool"
        )
        assert r["records_after_free"] == 0, f"rank {rank}: records left after free"
        assert r["pending_after_free"] == 0, f"rank {rank}: releases left queued"
        assert r["realloc_ok"], f"rank {rank}: realloc-after-free all-reduce wrong"
        assert r["infos_after_realloc"] == 1, (
            f"rank {rank}: {r['infos_after_realloc']} peer sets after realloc, want 1"
        )
        assert r["final_free_status"] == "freed", rank
