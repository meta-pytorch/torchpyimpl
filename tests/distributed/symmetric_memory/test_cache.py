# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rendezvous caching per (allocation, comm) on every backend."""

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

import torchpyimpl.distributed.symmetric_memory as symm
from torchpyimpl.distributed.symmetric_memory._symm_mem import (
    _cache_stats,
    buffer_offset,
)


_N_ELEMS = 1024


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(
    rank: int, world_size: int, master_port: int, result_path: str, backend: str
) -> None:
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
        symm.set_backend(backend)
        pg = dist.new_group(ranks=list(range(world_size)))

        t = symm.empty(_N_ELEMS, dtype=torch.float32, device=device)
        t.fill_(float(rank + 1))
        torch.cuda.synchronize(device)

        h0 = symm.rendezvous(t, pg)
        infos_after_first = _cache_stats()["peer_alloc_infos"]
        cache_identity = symm.rendezvous(t, pg) is h0
        repeat_no_new_peers = _cache_stats()["peer_alloc_infos"] == infos_after_first

        slice_off_elems = 128
        t_slice = t[slice_off_elems:]
        h_slice = symm.rendezvous(t_slice, pg)
        slice_reuses_peers = _cache_stats()["peer_alloc_infos"] == infos_after_first
        slice_offset_ok = h_slice.offset == slice_off_elems * 4
        slice_same_base = h_slice.buffer_ptrs[rank] == h0.buffer_ptrs[rank]

        pg2 = dist.new_group(ranks=list(range(world_size)))
        h_comm2 = symm.rendezvous(t, pg2)
        comm2_new_peers = _cache_stats()["peer_alloc_infos"] == infos_after_first + 1
        comm2_distinct = h_comm2 is not h0 and h_comm2.comm_ptr != h0.comm_ptr

        handles_before_free = _cache_stats()["handles"]
        signal_pad_ok = h0.buffer_ptr - h0.signal_pad_ptr == buffer_offset()
        del t, t_slice, h0, h_slice, h_comm2
        dist.barrier(group=pg)
        free_status = symm.free_pool(device, pg).value

        report: dict[str, Any] = {
            "rank": rank,
            "cache_identity": cache_identity,
            "repeat_no_new_peers": repeat_no_new_peers,
            "slice_reuses_peers": slice_reuses_peers,
            "slice_offset_ok": slice_offset_ok,
            "slice_same_base": slice_same_base,
            "comm2_new_peers": comm2_new_peers,
            "comm2_distinct": comm2_distinct,
            "handles_before_free": handles_before_free,
            "free_status": free_status,
            "infos_after_free": _cache_stats()["peer_alloc_infos"],
            "handles_after_free": _cache_stats()["handles"],
            "signal_pad_ok": signal_pad_ok,
        }
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, report, group=pg)
        if rank == 0:
            with open(result_path, "w") as f:
                json.dump({"reports": gathered}, f)
        dist.destroy_process_group(pg2)
        dist.destroy_process_group(pg)
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.peer_access
@pytest.mark.parametrize("backend", ["CUDA", "NCCL"])
def test_rendezvous_cache_per_allocation_and_comm(
    backend: str, tmp_path: pathlib.Path
) -> None:
    world_size = min(torch.cuda.device_count(), 8)
    result_path = str(tmp_path / "reports.json")
    mp.spawn(
        _worker,
        args=(world_size, _free_port(), result_path, backend),
        nprocs=world_size,
        join=True,
    )

    with open(result_path) as f:
        reports = json.load(f)["reports"]

    assert len(reports) == world_size
    for r in reports:
        rank = r["rank"]
        assert r["cache_identity"], f"rank {rank}: repeat rendezvous not cached"
        assert r["repeat_no_new_peers"], rank
        assert r["slice_reuses_peers"], f"rank {rank}: slice set up new peers"
        assert r["slice_offset_ok"], rank
        assert r["slice_same_base"], rank
        assert r["comm2_new_peers"], f"rank {rank}: second group reused peers"
        assert r["comm2_distinct"], rank
        assert r["handles_before_free"] == 3, rank
        assert r["free_status"] == "freed", f"rank {rank}: {r['free_status']}"
        assert r["infos_after_free"] == 0, f"rank {rank}: peer state after free"
        assert r["handles_after_free"] == 0, f"rank {rank}: handles after free"
        assert r["signal_pad_ok"], rank
