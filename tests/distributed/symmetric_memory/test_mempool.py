# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the pluggable allocator and the symmetric ``MemPool``."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import threading
from contextlib import nullcontext
from typing import Any

import pytest
import torch
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
    def __init__(self, pending: bool = False, skip_cleanup: bool = False) -> None:
        self.release_calls = 0
        self.pending = pending
        self.skip_cleanup = skip_cleanup

    def alloc(self, size: int, device: int) -> int:
        raise AssertionError("not expected")

    def free(self, ptr: int) -> None:
        raise AssertionError("not expected")

    def should_skip_cleanup(self, _device: int) -> bool:
        return self.skip_cleanup

    def deregister_windows(self, _device: int) -> int:
        return 0

    def has_unreleased_resources(self, _device: int) -> bool:
        return False

    def has_pending_releases(self, _device: int) -> bool:
        return self.pending

    def release_pending(self, _device: int) -> int:
        self.release_calls += 1
        return 0


class _CuMemBackend:
    """A PoolBackend over plain cuMemAlloc. Like the real backends, free only
    queues the segment and release_pending frees it."""

    def __init__(self) -> None:
        self.live: dict[int, int] = {}
        self.pending: list[int] = []
        self.released: list[int] = []

    def alloc(self, size: int, device: int) -> int:
        err, ptr = driver.cuMemAlloc(size)
        assert err == driver.CUresult.CUDA_SUCCESS, err
        self.live[int(ptr)] = size
        return int(ptr)

    def free(self, ptr: int) -> None:
        del self.live[ptr]
        self.pending.append(ptr)

    def should_skip_cleanup(self, _device: int) -> bool:
        return False

    def deregister_windows(self, _device: int) -> int:
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


def test_free_pool_keeps_pool_after_cuda_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:123")
    pool = _FakePool()
    backend = _FakeBackend(skip_cleanup=True)
    monkeypatch.setitem(_mempool._pools, 123, pool)
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)

    _mempool.free_pool(device)

    assert _mempool._pools[123] is pool


def test_free_pool_does_not_mask_empty_cache_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:125")
    pool = _FakePool()
    backend = _FakeBackend()
    monkeypatch.setitem(_mempool._pools, 125, pool)
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)
    monkeypatch.setattr(torch.cuda, "device", lambda _idx: nullcontext())

    def fail_empty_cache() -> None:
        raise RuntimeError("injected illegal memory access")

    monkeypatch.setattr(torch.cuda, "empty_cache", fail_empty_cache)

    _mempool.free_pool(device)

    assert 125 not in _mempool._pools
    assert backend.release_calls == 1


def test_free_pool_retries_release_after_pool_was_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:126")
    backend = _FakeBackend(pending=True)
    monkeypatch.setattr(_allocator, "_backend_resolver", lambda: backend)

    _mempool.free_pool(device)

    assert backend.release_calls == 1


def _mempool_worker(rank: int, result_path: str) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=device)  # ensure a live primary context
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

    del t, p1, p2
    _mempool.free_pool(device)

    report: dict[str, Any] = {
        "same_pool": same_pool,
        "routed": routed,
        "released": t_ptr in backend.released,
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
    assert r["released"], "free_pool did not release the segment"
    assert r["live_after_free"] == 0, "segments still live after free_pool"
    assert r["pending_after_free"] == 0, "queued releases not drained"


class _CountingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.builds = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "built symmetric CUDA allocator" in record.getMessage():
            self.builds += 1


def _concurrent_get_mem_pool(rank: int, result_path: str) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=device)  # ensure a live primary context
    backend = _CuMemBackend()
    _allocator.set_backend_resolver(lambda: backend)

    # _get_allocator is deliberately unsynchronized, so get_mem_pool's lock is
    # the only thing serializing the one-time ctypes build. Shrink the switch
    # interval so a regression there actually reproduces instead of being a
    # coin flip -- the build's one yield point is its logger.info, which the
    # counting handler below also makes actually write.
    counter = _CountingHandler()
    _allocator.logger.addHandler(counter)
    _allocator.logger.setLevel(logging.INFO)
    prev_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
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
    finally:
        sys.setswitchinterval(prev_interval)
        _allocator.logger.removeHandler(counter)

    report = {
        "n": len(pools),
        "distinct_pools": len({id(p) for p in pools}),
        "builds": counter.builds,
        "pinned_pairs": len(_allocator._pinned),
    }
    pools.clear()  # drop pool refs so free_pool can reclaim
    _mempool.free_pool(device)
    with open(result_path, "w") as f:
        json.dump(report, f)


@pytest.mark.gpu
def test_get_mem_pool_builds_allocator_once_under_concurrency() -> None:
    # get_mem_pool is the serialized entry point: concurrent callers must share
    # one pool, and the ctypes callback pair behind it must be built once. A
    # second build would waste an allocator (and, if _pinned were two slots
    # rather than append-only, orphan a live pair's raw addresses).
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
    assert r["builds"] == 1, (
        f"allocator built {r['builds']} times; get_mem_pool's lock is supposed "
        "to serialize the unsynchronized _get_allocator build"
    )
    assert r["pinned_pairs"] == 1, "callback pair was not pinned"
