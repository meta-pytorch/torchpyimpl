# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Python allocator and the symmetric ``MemPool``."""

from __future__ import annotations

import gc
import json
import tempfile
import threading
import weakref
from contextlib import nullcontext
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cuda.bindings import driver

from torchpyimpl.distributed.symmetric_memory import _allocator, _mempool


_N_ELEMS = 1024
_RACE_THREADS = 8


class _FakePool:
    def __init__(self, active_size: int = 0) -> None:
        self.active_size = active_size

    def snapshot(self) -> list[dict[str, int]]:
        return [{"active_size": self.active_size}]


class _FakeBackend:
    def __init__(
        self,
        pending: bool = False,
        skip_cleanup: bool = False,
        peer_error: bool = False,
        unreleased: bool = False,
        stuck: bool = False,
    ) -> None:
        self.release_calls = 0
        self.peer_calls = 0
        self.pending = pending
        self.skip_cleanup = skip_cleanup
        self.peer_error = peer_error
        self.unreleased = unreleased
        self.stuck = stuck

    def alloc(self, size: int, device: int, stream: int) -> int:
        raise AssertionError("not expected")

    def free(
        self,
        buffer_ptr: int,
        size: int,
        device: int,
        stream: int,
    ) -> None:
        raise AssertionError("not expected")

    def should_skip_cleanup(self, _device: int) -> bool:
        return self.skip_cleanup

    def release_peer_resources(self, _device: int) -> int:
        self.peer_calls += 1
        if self.peer_error:
            raise RuntimeError("injected peer cleanup failure")
        return 0

    def has_unreleased_resources(self, _device: int) -> bool:
        return self.unreleased

    def has_pending_releases(self, _device: int) -> bool:
        return self.pending

    def release_pending(self, _device: int) -> int:
        self.release_calls += 1
        if not self.stuck:
            self.pending = False
        return 0


class _CuMemBackend:
    def __init__(self) -> None:
        self.live: dict[int, int] = {}
        self.pending: list[int] = []
        self.released: list[int] = []
        self.devices: list[int] = []

    def alloc(self, size: int, device: int, stream: int) -> int:
        with torch.cuda.device(device):
            err, ptr = driver.cuMemAlloc(size)
        assert err == driver.CUresult.CUDA_SUCCESS, err
        self.live[int(ptr)] = size
        self.devices.append(device)
        return int(ptr)

    def free(
        self,
        buffer_ptr: int,
        size: int,
        device: int,
        stream: int,
    ) -> None:
        del self.live[buffer_ptr]
        self.pending.append(buffer_ptr)

    def should_skip_cleanup(self, _device: int) -> bool:
        return False

    def release_peer_resources(self, _device: int) -> int:
        return 0

    def has_unreleased_resources(self, _device: int) -> bool:
        return False

    def has_pending_releases(self, _device: int) -> bool:
        return bool(self.pending)

    def release_pending(self, _device: int) -> int:
        n = len(self.pending)
        for ptr in self.pending:
            (err,) = driver.cuMemFree(ptr)
            assert err == driver.CUresult.CUDA_SUCCESS, err
            self.released.append(ptr)
        self.pending.clear()
        return n


def test_get_backend_requires_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_allocator, "_backend_resolver", None)
    with pytest.raises(RuntimeError, match="no symmetric-memory backend"):
        _allocator.get_backend()


def test_get_mem_pool_uses_python_allocator_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:124")
    backend = _FakeBackend()
    alloc_calls: list[tuple[int, int, int]] = []
    free_calls: list[tuple[int, int, int, int]] = []
    callbacks: dict[str, Any] = {}

    def backend_alloc(size: int, device_idx: int, stream: int) -> int:
        alloc_calls.append((size, device_idx, stream))
        return 0x1234

    def backend_free(ptr: int, size: int, device_idx: int, stream: int) -> None:
        free_calls.append((ptr, size, device_idx, stream))

    backend.alloc = backend_alloc  # type: ignore[method-assign]
    backend.free = backend_free  # type: ignore[method-assign]

    class FakeMemPool:
        @classmethod
        def from_py_allocator(
            cls,
            alloc_fn: Any,
            free_fn: Any,
            **options: Any,
        ) -> FakeMemPool:
            callbacks.update(alloc=alloc_fn, free=free_fn, options=options)
            return cls()

    current: list[int] = []

    def fake_device(idx: int) -> Any:
        current.append(idx)
        return nullcontext()

    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)
    monkeypatch.setattr(_mempool, "_pools", {})
    monkeypatch.setattr(torch.cuda, "MemPool", FakeMemPool)
    monkeypatch.setattr(torch.cuda, "device", fake_device)

    pool = _mempool.get_mem_pool(device)

    assert isinstance(pool, FakeMemPool)
    assert current == [124], "the pool was not created on the requested device"
    assert callbacks["options"] == {"use_on_oom": False, "no_split": True}
    assert callbacks["alloc"](64, 124, 0x5678) == 0x1234
    callbacks["free"](0x1234, 64, 124, 0x5678)
    assert alloc_calls == [(64, 124, 0x5678)]
    assert free_calls == [(0x1234, 64, 124, 0x5678)]


def test_free_pool_keeps_pool_after_cuda_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:123")
    pool = _FakePool()
    backend = _FakeBackend(skip_cleanup=True)
    monkeypatch.setitem(_mempool._pools, 123, pool)
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)

    status = _mempool.free_pool(device)

    assert status is _mempool.FreePoolStatus.CUDA_ERROR
    assert _mempool._pools[123] is pool


def test_free_pool_does_not_mask_empty_cache_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:125")
    backend = _FakeBackend()
    monkeypatch.setitem(_mempool._pools, 125, _FakePool())
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)
    monkeypatch.setattr(torch.cuda, "device", lambda _idx: nullcontext())

    def fail_empty_cache() -> None:
        raise RuntimeError("injected illegal memory access")

    monkeypatch.setattr(torch.cuda, "empty_cache", fail_empty_cache)

    status = _mempool.free_pool(device)

    assert status is _mempool.FreePoolStatus.FREED
    assert 125 not in _mempool._pools
    assert backend.release_calls == 1


def test_free_pool_retries_release_after_pool_was_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:126")
    backend = _FakeBackend(pending=True)
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)

    status = _mempool.free_pool(device)

    assert status is _mempool.FreePoolStatus.FREED
    assert backend.release_calls == 1


def test_free_pool_reports_nothing_to_free(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeBackend()
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)

    status = _mempool.free_pool(torch.device("cuda:127"))

    assert status is _mempool.FreePoolStatus.NOTHING_TO_FREE
    assert backend.release_calls == 0


def test_free_pool_reports_releases_left_in_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend(pending=True, stuck=True)
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)

    status = _mempool.free_pool(torch.device("cuda:122"))

    assert status is _mempool.FreePoolStatus.INCOMPLETE
    assert backend.release_calls == 1


@pytest.mark.parametrize(
    "failure", [{"peer_error": True}, {"unreleased": True}], ids=["raises", "left"]
)
def test_free_pool_keeps_pool_after_incomplete_peer_cleanup(
    monkeypatch: pytest.MonkeyPatch, failure: dict[str, bool]
) -> None:
    backend = _FakeBackend(**failure)
    monkeypatch.setitem(_mempool._pools, 119, _FakePool())
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)

    status = _mempool.free_pool(torch.device("cuda:119"))

    assert status is _mempool.FreePoolStatus.INCOMPLETE
    assert 119 in _mempool._pools
    assert backend.peer_calls == 1
    assert backend.release_calls == 0


def test_free_pool_refuses_pool_with_outside_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _FakePool()
    backend = _FakeBackend()
    monkeypatch.setitem(_mempool._pools, 120, pool)
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)

    status = _mempool.free_pool(torch.device("cuda:120"))

    assert status is _mempool.FreePoolStatus.IN_USE
    assert _mempool._pools[120] is pool
    assert backend.peer_calls == 0


def test_free_pool_refuses_pool_with_live_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    monkeypatch.setitem(_mempool._pools, 121, _FakePool(active_size=4096))
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)

    status = _mempool.free_pool(torch.device("cuda:121"))

    assert status is _mempool.FreePoolStatus.IN_USE
    assert 121 in _mempool._pools
    assert backend.peer_calls == 0


def _two_rank_group(monkeypatch: pytest.MonkeyPatch, peer: list[str | None]) -> Any:
    votes = iter(peer)

    def all_gather_object(out: list[object], obj: object, group: object) -> None:
        out[0], out[1] = obj, next(votes)

    monkeypatch.setattr(_mempool.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(_mempool.dist, "all_gather_object", all_gather_object)
    return cast(dist.ProcessGroup, object())


def test_free_pool_with_group_keeps_pool_if_a_peer_is_in_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    monkeypatch.setitem(_mempool._pools, 116, _FakePool())
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)
    group = _two_rank_group(monkeypatch, ["in_use"])

    status = _mempool.free_pool(torch.device("cuda:116"), group)

    assert status is _mempool.FreePoolStatus.IN_USE
    assert 116 in _mempool._pools
    assert backend.peer_calls == 0


def test_free_pool_with_group_reports_a_peer_incomplete_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    monkeypatch.setitem(_mempool._pools, 115, _FakePool())
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)
    monkeypatch.setattr(torch.cuda, "device", lambda _idx: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    group = _two_rank_group(monkeypatch, [None, "incomplete"])

    status = _mempool.free_pool(torch.device("cuda:115"), group)

    assert status is _mempool.FreePoolStatus.INCOMPLETE
    assert 115 not in _mempool._pools
    assert backend.peer_calls == 1


def _mempool_worker(rank: int, result_path: str) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=device)
    backend = _CuMemBackend()
    _allocator.set_backend_resolver(lambda: backend)

    p1 = _mempool.get_mem_pool(device)
    p2 = _mempool.get_mem_pool(device)
    same_pool = p1 is p2

    with torch.cuda.use_mem_pool(p1):
        t = torch.empty(_N_ELEMS, dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    t_ptr = int(t.data_ptr())
    routed = t_ptr in backend.live
    pool_ref = weakref.ref(p1)

    del t, p1, p2
    status = _mempool.free_pool(device)
    gc.collect()

    report: dict[str, Any] = {
        "same_pool": same_pool,
        "routed": routed,
        "status": status.value,
        "released": t_ptr in backend.released,
        "pool_collected": pool_ref() is None,
        "live_after_free": len(backend.live),
        "pending_after_free": len(backend.pending),
    }
    with open(result_path, "w") as f:
        json.dump(report, f)


@pytest.mark.gpu
def test_mem_pool_routes_to_backend_and_frees() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        result_path = tmp.name
    mp.spawn(_mempool_worker, args=(result_path,), nprocs=1, join=True)

    with open(result_path) as f:
        r = json.load(f)
    assert r["same_pool"], "get_mem_pool is not memoized per device"
    assert r["routed"], "pool allocation did not come from the backend"
    assert r["status"] == "freed", r["status"]
    assert r["released"], "free_pool did not release the segment"
    assert r["pool_collected"], "allocator callbacks retained the MemPool"
    assert r["live_after_free"] == 0, "segments still live after free_pool"
    assert r["pending_after_free"] == 0, "queued releases not drained"


def _in_use_worker(rank: int, result_path: str) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=device)
    backend = _CuMemBackend()
    _allocator.set_backend_resolver(lambda: backend)

    with torch.cuda.use_mem_pool(_mempool.get_mem_pool(device)):
        t = torch.empty(_N_ELEMS, dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    busy = _mempool.free_pool(device)
    kept = 0 in _mempool._pools and int(t.data_ptr()) in backend.live

    del t
    idle = _mempool.free_pool(device)

    report: dict[str, Any] = {
        "busy": busy.value,
        "kept": kept,
        "idle": idle.value,
        "live_after_free": len(backend.live),
        "pending_after_free": len(backend.pending),
    }
    with open(result_path, "w") as f:
        json.dump(report, f)


@pytest.mark.gpu
def test_free_pool_keeps_pool_while_tensors_are_alive() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        result_path = tmp.name
    mp.spawn(_in_use_worker, args=(result_path,), nprocs=1, join=True)

    with open(result_path) as f:
        r = json.load(f)
    assert r["busy"] == "in_use", r["busy"]
    assert r["kept"], "free_pool dropped a pool that a live tensor uses"
    assert r["idle"] == "freed", r["idle"]
    assert r["live_after_free"] == 0, "segments still live after free_pool"
    assert r["pending_after_free"] == 0, "queued releases not drained"


def _other_device_worker(rank: int, result_path: str) -> None:
    torch.cuda.set_device(0)
    _ = torch.empty(1, device="cuda:0")
    backend = _CuMemBackend()
    _allocator.set_backend_resolver(lambda: backend)

    device = torch.device("cuda:1")
    with torch.cuda.use_mem_pool(_mempool.get_mem_pool(device), device=device):
        t = torch.empty(_N_ELEMS, dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)

    report: dict[str, Any] = {
        "routed": int(t.data_ptr()) in backend.live,
        "devices": backend.devices,
    }
    del t
    report["status"] = _mempool.free_pool(device).value
    with open(result_path, "w") as f:
        json.dump(report, f)


@pytest.mark.multi_gpu
def test_get_mem_pool_uses_requested_device() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        result_path = tmp.name
    mp.spawn(_other_device_worker, args=(result_path,), nprocs=1, join=True)

    with open(result_path) as f:
        r = json.load(f)
    assert r["routed"], "cuda:1 allocation did not come from the backend"
    assert r["devices"] == [1], r["devices"]
    assert r["status"] == "freed", r["status"]


def _concurrent_get_mem_pool(rank: int, result_path: str) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=device)
    backend = _CuMemBackend()
    _allocator.set_backend_resolver(lambda: backend)

    pools: list[Any] = []
    start = threading.Barrier(_RACE_THREADS)

    def run() -> None:
        start.wait()
        pools.append(_mempool.get_mem_pool(device))

    threads = [threading.Thread(target=run) for _ in range(_RACE_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    report = {
        "n": len(pools),
        "distinct_pools": len({id(p) for p in pools}),
    }
    pools.clear()
    _mempool.free_pool(device)
    with open(result_path, "w") as f:
        json.dump(report, f)


@pytest.mark.gpu
def test_get_mem_pool_builds_pool_once_under_concurrency() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        result_path = tmp.name
    mp.spawn(_concurrent_get_mem_pool, args=(result_path,), nprocs=1, join=True)

    with open(result_path) as f:
        r = json.load(f)
    assert r["n"] == _RACE_THREADS
    assert r["distinct_pools"] == 1, (
        f"{r['distinct_pools']} pools created for one device; get_mem_pool's "
        "lock is not serializing pool creation"
    )
