# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import socket
from contextlib import closing, nullcontext
from typing import Any, cast, TYPE_CHECKING


if TYPE_CHECKING:
    import pathlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _dist_worker import dist_worker

import torchpyimpl.distributed.symmetric_memory._symm_mem as sm
from torchpyimpl.distributed.symmetric_memory._comm import get_group_info
from torchpyimpl.distributed.symmetric_memory.backends import _cuda_ipc


def test_close_peers_tears_down_multicast_before_peer_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    mapping = _cuda_ipc._CudaMapping(0x1000, 4096, object(), 0)
    mc = _cuda_ipc._CudaMulticast(handle=object(), addr=0x9000, size=4096, device=0)
    state = _cuda_ipc._CudaPeerState(imported=[mapping], multicast=mc)
    empty = torch.empty(0, dtype=torch.int64)
    pai = sm._PeerAllocInfo(state, [], [], empty, empty, 0, device=0)
    monkeypatch.setattr(
        _cuda_ipc._cuda, "should_skip_cuda_cleanup", lambda _device: False
    )
    monkeypatch.setattr(_cuda_ipc.torch.cuda, "device", lambda _idx: nullcontext())
    monkeypatch.setattr(
        _cuda_ipc.CUDAIPCAllocator,
        "_teardown_multicast",
        staticmethod(lambda *_args: calls.append("multicast") or True),
    )
    monkeypatch.setattr(
        _cuda_ipc, "cu_vmm_free", lambda *_args: calls.append("peer") or True
    )

    assert _cuda_ipc.CUDAIPCAllocator().close_peers(pai) is sm.CleanupStatus.CLEANED
    assert calls == ["multicast", "peer"]
    assert pai.window is None


@pytest.mark.parametrize(
    ("world_size", "support"),
    [(1, [True]), (2, [True, False]), (2, [False, False])],
)
def test_init_multicast_skips_without_unanimous_support(
    monkeypatch: pytest.MonkeyPatch, world_size: int, support: list[bool]
) -> None:
    monkeypatch.setattr(
        _cuda_ipc, "_all_agree", lambda *_args: pytest.fail("collective ran")
    )
    mine = _cuda_ipc._CudaMapping(0x100000, 4096, object(), 0)
    gathered = [{"multicast": s} for s in support]
    result = _cuda_ipc.CUDAIPCAllocator()._init_multicast(
        mine,
        cast(list[dict[str, object]], gathered),
        cast(dist.ProcessGroup, object()),
        0,
        world_size,
        0,
        sm.buffer_offset(),
        _cuda_ipc.IpcHandleType.FABRIC,
    )
    assert result == (None, 0)


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dist_worker
def _worker(rank: int, world_size: int, out: str, transport: str) -> None:
    device = torch.device(f"cuda:{rank}")
    os.environ["TORCHPYIMPL_CUDA_IPC_TRANSPORT"] = transport
    _cuda_ipc.reset_transport()
    pg = dist.new_group(ranks=list(range(world_size)))
    comm_ptr = get_group_info(pg, device, require_comm=False).comm_ptr
    nbytes = 4096
    buf = sm.alloc(nbytes, device)
    h = sm.rendezvous(buf, nbytes, comm_ptr, rank, world_size, device, group=pg)
    report: dict[str, Any] = {
        "rank": rank,
        "supported": _cuda_ipc.cu_device_supports_multicast(rank),
        "has_multicast": h.multicast_ptr != 0,
        "offset_ok": h.multicast_ptr - h.multicast_signal_pad_ptr == sm.buffer_offset(),
    }
    dist.barrier(group=pg)
    sm.free(buf)
    report["peer_alloc_infos_after_free"] = sm._cache_stats()["peer_alloc_infos"]
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, report, group=pg)
    if rank == 0:
        with open(out, "w") as f:
            json.dump(gathered, f)
    dist.destroy_process_group(pg)


@pytest.mark.multi_gpu
@pytest.mark.peer_access
@pytest.mark.parametrize(
    "transport", [pytest.param("fabric", marks=pytest.mark.fabric), "fd"]
)
def test_multicast_setup_and_teardown(transport: str, tmp_path: pathlib.Path) -> None:
    if transport == "fd" and not _cuda_ipc.cu_device_supports_posix_fd(0):
        pytest.skip("no POSIX-fd handle support on this device")
    world_size = min(torch.cuda.device_count(), 8)
    out = str(tmp_path / "reports.json")
    mp.spawn(
        _worker, args=(world_size, _free_port(), out, transport), nprocs=world_size
    )
    with open(out) as f:
        reports = json.load(f)

    assert len(reports) == world_size
    expected = all(r["supported"] for r in reports)
    if not expected:
        pytest.skip("devices do not support multicast")
    for r in reports:
        rank = r["rank"]
        assert r["has_multicast"], f"rank {rank}: no multicast pointer"
        assert r["offset_ok"], f"rank {rank}: multicast_ptr not at the data buffer"
        assert r["peer_alloc_infos_after_free"] == 0, rank
