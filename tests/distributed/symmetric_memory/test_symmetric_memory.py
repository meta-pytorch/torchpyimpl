# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end symmetric memory tests on the CUDA-IPC backend."""

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

import torchpyimpl.distributed.symmetric_memory as symm
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


# ── MemPool API ───────────────────────────────────────────────────────────────


def _worker(rank: int, world_size: int, port: int, out: str, transport: str) -> None:
    # Must be set before the first allocation: cuMemCreate bakes the handle type
    # into the block.
    os.environ["TORCHPYIMPL_CUDA_IPC_TRANSPORT"] = transport
    _cuda_ipc.reset_transport()
    device = _init_dist(rank, world_size, port)
    try:
        symm.set_backend("CUDA")
        pg = dist.new_group(ranks=list(range(world_size)))

        tensors = [
            symm.empty(_N_ELEMS, dtype=torch.float32, device=device)
            for _ in range(_NUM_TENSORS)
        ]
        for t in tensors:
            t.fill_(float(rank + 1))
        torch.cuda.synchronize(device)

        # Rendezvous in reverse order: each call establishes only its own
        # covering allocation.
        handles = [symm.rendezvous(t, pg) for t in reversed(tensors)]
        handles.reverse()
        hdl = handles[0]

        stats_before = sm._cache_stats()
        cache_same = symm.rendezvous(tensors[0], pg) is hdl
        stats_after = sm._cache_stats()

        dist.barrier(group=pg)
        acc = torch.zeros(_N_ELEMS, dtype=torch.float32, device=device)
        for peer in range(world_size):
            acc += symm.get_buffer(hdl, peer, (_N_ELEMS,), torch.float32)
        torch.cuda.synchronize(device)

        pad = symm.get_signal_pad(hdl, rank)
        torch.cuda.synchronize(device)

        mc_flags: list[Any] = [None] * world_size
        dist.all_gather_object(mc_flags, symm.has_multicast_support(hdl), group=pg)

        report: dict[str, Any] = {
            "rank": rank,
            "transport": _cuda_ipc.get_transport(rank).value,
            "buffer_ptrs_len": [len(h.buffer_ptrs) for h in handles],
            "all_ptrs_nonzero": all(
                all(p != 0 for p in h.buffer_ptrs) for h in handles
            ),
            "distinct_bases": len({h.buffer_ptrs[rank] for h in handles}),
            "peer_alloc_infos": stats_after["peer_alloc_infos"],
            "peer_alloc_infos_unchanged": (
                stats_before["peer_alloc_infos"] == stats_after["peer_alloc_infos"]
            ),
            "cache_same": cache_same,
            "signal_pad_ok": hdl.buffer_ptr - hdl.signal_pad_ptr == sm.buffer_offset(),
            "pad_zeroed": bool(torch.all(pad == 0).item()),
            "sum_first": float(acc[0].item()),
            "sum_all_equal": bool(torch.all(acc == acc[0]).item()),
            "multicast_agreed": len(set(mc_flags)) == 1,
        }

        dist.barrier(group=pg)
        del tensors, t
        report["free_status"] = symm.free_pool(device).value
        report["peer_alloc_infos_after_free"] = sm._cache_stats()["peer_alloc_infos"]
        try:
            symm.get_buffer(hdl, rank, (1,), torch.float32)
            report["stale_handle_rejected"] = False
        except RuntimeError:
            report["stale_handle_rejected"] = True
        _gather_reports(report, pg, out)
        dist.destroy_process_group(pg)
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.peer_access
@pytest.mark.parametrize(
    "transport", [pytest.param("fabric", marks=pytest.mark.fabric), "fd"]
)
def test_symm_mem_mempool(transport: str, tmp_path: pathlib.Path) -> None:
    if transport == "fd" and not _cuda_ipc.cu_device_supports_posix_fd(0):
        pytest.skip("no POSIX-fd handle support on this device")
    world_size = min(torch.cuda.device_count(), 8)
    reports = _spawn(_worker, world_size, str(tmp_path / "reports.json"), transport)

    expected_sum = float(sum(range(1, world_size + 1)))
    assert len(reports) == world_size
    for r in reports:
        rank = r["rank"]
        assert r["transport"] == transport, rank
        assert all(n == world_size for n in r["buffer_ptrs_len"]), rank
        assert r["all_ptrs_nonzero"], rank
        # no_split => one segment per tensor => distinct bases, one peer set each
        assert r["distinct_bases"] == _NUM_TENSORS, rank
        assert r["peer_alloc_infos"] == _NUM_TENSORS, rank
        assert r["peer_alloc_infos_unchanged"], rank
        assert r["cache_same"], f"rank {rank}: repeat rendezvous returned a new handle"
        assert r["signal_pad_ok"], rank
        assert r["pad_zeroed"], f"rank {rank}: signal pad not zeroed at allocation"
        assert r["sum_all_equal"], rank
        assert r["sum_first"] == expected_sum, (
            f"rank {rank}: cross-rank sum = {r['sum_first']}, want {expected_sum}"
        )
        assert r["multicast_agreed"], f"rank {rank}: ranks disagree on multicast"
        assert r["peer_alloc_infos_after_free"] == 0, (
            f"rank {rank}: peer state still registered after free_pool"
        )
        assert r["free_status"] == "freed", rank
        assert r["stale_handle_rejected"], f"rank {rank}: handle used after free_pool"


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def _fd_leak_worker(rank: int, world_size: int, port: int, out: str) -> None:
    """Repeat alloc -> rendezvous -> free_pool and watch the fd table. Each
    rendezvous on the fd transport exports one fd, receives world_size - 1 over
    SCM_RIGHTS, and opens a Unix socket; any escape grows the count per round."""
    os.environ["TORCHPYIMPL_CUDA_IPC_TRANSPORT"] = "fd"
    _cuda_ipc.reset_transport()
    device = _init_dist(rank, world_size, port)
    try:
        pg = dist.new_group(ranks=list(range(world_size)))
        counts = []
        for _ in range(4):
            t = symm.empty(_N_ELEMS, dtype=torch.float32, device=device)
            symm.rendezvous(t, pg)
            del t
            symm.free_pool(device)
            dist.barrier(group=pg)
            counts.append(_open_fds())
        _gather_reports({"rank": rank, "counts": counts}, pg, out)
        dist.destroy_process_group(pg)
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.peer_access
def test_posix_fd_transport_does_not_leak_fds(tmp_path: pathlib.Path) -> None:
    if not _cuda_ipc.cu_device_supports_posix_fd(0):
        pytest.skip("no POSIX-fd handle support on this device")
    world_size = min(torch.cuda.device_count(), 8)
    reports = _spawn(_fd_leak_worker, world_size, str(tmp_path / "reports.json"))
    assert len(reports) == world_size
    for r in reports:
        # The first round warms up lazily-opened driver/NCCL fds; after that a
        # leaking transport grows every round.
        counts = r["counts"]
        assert counts[-1] <= counts[-2], (
            f"rank {r['rank']}: open fd count grew across rounds ({counts})"
        )


def _sliced_accessor_worker(rank: int, world_size: int, port: int, out: str) -> None:
    """A handle from a slice: ``offset`` is measured from the allocation base
    but ``size`` is the view's, so bounds checks must not count the offset
    twice."""
    device = _init_dist(rank, world_size, port)
    try:
        pg = dist.new_group(ranks=list(range(world_size)))
        base = symm.empty(_N_ELEMS, dtype=torch.float32, device=device)
        hdl = symm.rendezvous(base[_N_ELEMS // 2 :], pg)
        r: dict[str, Any] = {"rank": rank, "offset": hdl.offset}
        fits = hdl.size // 4

        symm.get_remote_tensor(hdl, rank, (fits,), torch.float32)
        r["fills_to_end_accepted"] = True
        try:
            symm.get_remote_tensor(hdl, rank, (fits + 1,), torch.float32)
            r["past_end_rejected"] = False
        except ValueError:
            r["past_end_rejected"] = True
        try:
            symm.get_buffer(hdl, -1, (1,), torch.float32)
            r["negative_peer_rejected"] = False
        except ValueError:
            r["negative_peer_rejected"] = True
        r["device_ok"] = (
            symm.get_buffer(hdl, rank, (1,), torch.float32).device == device
        )
        try:
            symm.get_buffer(hdl, rank, (1,), torch.float32, storage_offset=-1)
            r["negative_offset_rejected"] = False
        except ValueError:
            r["negative_offset_rejected"] = True
        r["group_name_ok"] = (
            symm.rendezvous(base[_N_ELEMS // 2 :], pg.group_name) is hdl
        )
        pad = symm.get_signal_pad(hdl, rank, sizes=[], dtype=None)
        r["pad_defaults_ok"] = pad.numel() * 4 == sm.signal_pad_bytes()
        _gather_reports(r, pg, out)
        dist.destroy_process_group(pg)
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.peer_access
def test_sliced_handle_accessors(tmp_path: pathlib.Path) -> None:
    world_size = min(torch.cuda.device_count(), 8)
    reports = _spawn(
        _sliced_accessor_worker, world_size, str(tmp_path / "reports.json")
    )
    assert len(reports) == world_size
    for r in reports:
        rank = r["rank"]
        assert r["offset"] > 0, f"rank {rank}: slice did not produce a view offset"
        assert r["fills_to_end_accepted"], rank
        assert r["past_end_rejected"], f"rank {rank}: past-the-end request accepted"
        assert r["negative_peer_rejected"], f"rank {rank}: negative peer accepted"
        assert r["device_ok"], f"rank {rank}: view labelled with the wrong device"
        assert r["negative_offset_rejected"], f"rank {rank}: negative offset accepted"
        assert r["group_name_ok"], f"rank {rank}: rendezvous by group name failed"
        assert r["pad_defaults_ok"], f"rank {rank}: wrong default signal pad view"


def _empty_device_worker(rank: int, result_path: str) -> None:
    os.environ["TORCHPYIMPL_CUDA_IPC_TRANSPORT"] = "fd"
    torch.cuda.set_device(0)
    t = symm.empty(_N_ELEMS, device=torch.device("cuda:1"))
    rec, _ = sm.find_covering(t.data_ptr())
    with open(result_path, "w") as f:
        json.dump({"device": str(t.device), "symmetric": rec is not None}, f)


@pytest.mark.multi_gpu
def test_empty_uses_requested_device(tmp_path: pathlib.Path) -> None:
    # cuda:0 is current; the tensor must still come from cuda:1's pool.
    result_path = str(tmp_path / "report.json")
    mp.spawn(_empty_device_worker, args=(result_path,), nprocs=1)
    with open(result_path) as f:
        assert json.load(f) == {"device": "cuda:1", "symmetric": True}
