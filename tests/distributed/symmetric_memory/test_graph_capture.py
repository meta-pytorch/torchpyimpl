# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import socket
from contextlib import closing
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torchpyimpl.distributed.symmetric_memory as symm
import torchpyimpl.distributed.symmetric_memory._symm_mem as sm


_N = 1024
_REPLAYS = 3


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init(rank: int, world_size: int, port: int, backend: str) -> torch.device:
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
    symm.set_backend(backend)
    return device


def _rejected_in_capture(fn: Callable[[], object], stream: torch.cuda.Stream) -> str:
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            fn()
    except RuntimeError as e:
        return str(e)
    return ""


def _capture_worker(
    rank: int, world_size: int, port: int, out: str, backend: str
) -> None:
    device = _init(rank, world_size, port, backend)
    pg = dist.group.WORLD
    peer = (rank + 1) % world_size
    t = symm.empty(_N, dtype=torch.float32, device=device)
    fresh = symm.empty(_N, dtype=torch.float32, device=device)
    hdl = symm.rendezvous(t, pg)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        hdl.barrier()
    torch.cuda.synchronize()

    r: dict[str, Any] = {"rank": rank}
    r["alloc"] = _rejected_in_capture(
        lambda: symm.empty(1 << 22, device=device), stream
    )
    r["standalone_alloc"] = _rejected_in_capture(lambda: sm.alloc(4096, device), stream)
    r["first_rendezvous"] = _rejected_in_capture(
        lambda: symm.rendezvous(fresh, pg), stream
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        cached = symm.rendezvous(t, pg)
        view = symm.rendezvous(t[_N // 2 :], pg)
        t.fill_(rank + 100.0)
        cached.barrier()
    r["cached_is_same"] = cached is hdl
    r["view_offset_ok"] = view.offset == _N // 2 * 4
    for _ in range(_REPLAYS):
        graph.replay()
    torch.cuda.synchronize()
    got = symm.get_buffer(hdl, peer, (_N,), torch.float32)
    r["replayed_peer_value_ok"] = bool((got == peer + 100.0).all())

    symm.rendezvous(fresh, pg).barrier()
    torch.cuda.synchronize()
    r["fresh_rendezvous_after"] = True

    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, r, group=pg)
    if rank == 0:
        with open(out, "w") as f:
            json.dump({"reports": gathered}, f)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.peer_access
@pytest.mark.parametrize("backend", ["CUDA", "NCCL"])
def test_capture_rejects_setup_and_replays_cached_handles(
    backend: str, tmp_path: pathlib.Path
) -> None:
    world_size = min(torch.cuda.device_count(), 8)
    out = str(tmp_path / "reports.json")
    mp.spawn(
        _capture_worker,
        args=(world_size, _free_port(), out, backend),
        nprocs=world_size,
    )
    with open(out) as f:
        reports = json.load(f)["reports"]
    assert len(reports) == world_size
    for r in reports:
        rank = r["rank"]
        for op in ("alloc", "standalone_alloc", "first_rendezvous"):
            assert "capturing a graph" in r[op], f"rank {rank}: {op} gave {r[op]!r}"
        assert "before starting capture" in r["first_rendezvous"], rank
        assert r["cached_is_same"], f"rank {rank}: cached rendezvous built a new handle"
        assert r["view_offset_ok"], f"rank {rank}: wrong view offset in capture"
        assert r["replayed_peer_value_ok"], f"rank {rank}: replay saw a stale peer"
        assert r["fresh_rendezvous_after"], rank
