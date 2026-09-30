# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the backend-agnostic allocator in ``symmetric_memory._symm_mem``."""

from __future__ import annotations

import os

import pytest
import torch

import torchpyimpl.distributed.symmetric_memory._symm_mem as sm


def test_buffer_offset_matches_signal_pad_rounding() -> None:
    # buffer_offset is the signal-pad size rounded up to the 128-B alignment.
    assert sm.buffer_offset() == sm._round_up(
        sm.signal_pad_bytes(), sm.SIGNAL_PAD_ALIGNMENT
    )
    # The torch default pad (32 * 72 * 4 = 9216) is already 128-aligned.
    assert sm.buffer_offset() == sm.signal_pad_bytes()


def test_signal_pad_env_rejects_nonpositive() -> None:
    # set_signal_pad_size rejects <= 0, but the env var reaches buffer_offset
    # without going through it, so the read has to validate too.
    prev = os.environ.get(sm._SIGNAL_PAD_ENV)
    try:
        for bad in ("0", "-128"):
            os.environ[sm._SIGNAL_PAD_ENV] = bad
            with pytest.raises(ValueError):
                sm.signal_pad_bytes()
            with pytest.raises(ValueError):
                sm.buffer_offset()
    finally:
        if prev is None:
            os.environ.pop(sm._SIGNAL_PAD_ENV, None)
        else:
            os.environ[sm._SIGNAL_PAD_ENV] = prev


_NBYTES = 4001  # non-16-aligned user size -> round_up(., 16) = 4016


class _FaultInjectingAllocator(sm.SymmetricMemoryAllocatorBase):
    def __init__(self) -> None:
        super().__init__()
        self.free_statuses: list[sm.CleanupStatus] = []
        self.close_statuses: dict[int, list[sm.CleanupStatus]] = {}
        self.free_calls = 0
        self.close_calls: list[int] = []

    def alloc_physical(self, block_size: int, device: int) -> tuple[int, object]:
        return 0x100000, object()

    def free_physical(self, handle: object) -> sm.CleanupStatus:
        self.free_calls += 1
        if self.free_statuses:
            return self.free_statuses.pop(0)
        return sm.CleanupStatus.CLEANED

    def allocation_base(self, ptr: int) -> int:
        return 0x100000

    def establish_peers(
        self,
        rec: sm.AllocRecord,
        comm_ptr: int,
        world_size: int,
        device: torch.device,
        group: object | None = None,
    ) -> sm._PeerAllocInfo:
        data_ptr = rec.alloc_base + sm.buffer_offset()
        ptrs = torch.tensor([data_ptr], dtype=torch.int64)
        pads = torch.tensor([rec.alloc_base], dtype=torch.int64)
        return sm._PeerAllocInfo(
            window=comm_ptr,
            buffers=[data_ptr],
            signal_pads=[rec.alloc_base],
            buffers_dev=ptrs,
            signal_pads_dev=pads,
            multicast_ptr=0,
            device=rec.device,
        )

    def close_peers(self, pai: sm._PeerAllocInfo) -> sm.CleanupStatus:
        comm_ptr = int(pai.window)
        self.close_calls.append(comm_ptr)
        statuses = self.close_statuses.get(comm_ptr)
        if statuses:
            status = statuses.pop(0)
        else:
            status = sm.CleanupStatus.CLEANED
        if status is not sm.CleanupStatus.DEFERRED:
            pai.window = None
        return status


def test_alloc_rolls_back_if_publication_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = _FaultInjectingAllocator()
    monkeypatch.setattr(sm, "_pad_frozen", False)

    def fail_memset(*_args: object) -> None:
        raise torch.OutOfMemoryError("injected")

    monkeypatch.setattr(sm, "cu_memset_d8", fail_memset)

    with pytest.raises(torch.OutOfMemoryError, match="injected"):
        allocator.alloc(1024, 0)

    assert allocator.free_calls == 1
    assert allocator.cache_stats()["records"] == 0


def test_alloc_retries_deferred_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    allocator = _FaultInjectingAllocator()
    allocator.free_statuses = [
        sm.CleanupStatus.DEFERRED,
        sm.CleanupStatus.CLEANED,
    ]
    monkeypatch.setattr(sm, "_pad_frozen", False)

    def fail_memset(*_args: object) -> None:
        raise torch.OutOfMemoryError("injected")

    monkeypatch.setattr(sm, "cu_memset_d8", fail_memset)

    with pytest.raises(torch.OutOfMemoryError, match="injected"):
        allocator.alloc(1024, 0)

    assert allocator.cache_stats()["pending"] == 1
    assert allocator.release_pending() == 1
    assert allocator.cache_stats()["pending"] == 0


def test_release_pending_retries_deferred_window_and_drains_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = _FaultInjectingAllocator()
    monkeypatch.setattr(sm, "_pad_frozen", False)
    monkeypatch.setattr(sm, "cu_memset_d8", lambda *_args: None)
    ptr = allocator.alloc(1024, 0)
    device = torch.device("cpu")
    allocator.rendezvous(ptr, 11, 0, 1, device)
    allocator.rendezvous(ptr, 12, 0, 1, device)
    allocator.close_statuses = {
        11: [sm.CleanupStatus.DEFERRED, sm.CleanupStatus.CLEANED]
    }

    allocator.free(ptr)
    assert allocator.release_pending() == 0
    assert allocator.close_calls == [11, 12]
    assert allocator.free_calls == 0
    assert allocator.cache_stats()["pending"] == 1

    assert allocator.release_pending() == 1
    assert allocator.close_calls == [11, 12, 11]
    assert allocator.free_calls == 1
    assert allocator.cache_stats()["pending"] == 0


def test_release_pending_does_not_count_deferred_physical_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = _FaultInjectingAllocator()
    monkeypatch.setattr(sm, "_pad_frozen", False)
    allocator.free_statuses = [
        sm.CleanupStatus.DEFERRED,
        sm.CleanupStatus.CLEANED,
    ]
    monkeypatch.setattr(sm, "cu_memset_d8", lambda *_args: None)
    ptr = allocator.alloc(1024, 0)

    allocator.free(ptr)
    assert allocator.release_pending() == 0
    assert allocator.cache_stats()["pending"] == 1
    assert allocator.release_pending() == 1
    assert allocator.cache_stats()["pending"] == 0


def test_release_pending_retains_failed_record_and_drains_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = _FaultInjectingAllocator()
    monkeypatch.setattr(sm, "_pad_frozen", False)
    monkeypatch.setattr(sm, "cu_memset_d8", lambda *_args: None)
    ptr = allocator.alloc(1024, 0)
    device = torch.device("cpu")
    allocator.rendezvous(ptr, 11, 0, 1, device)
    allocator.rendezvous(ptr, 12, 0, 1, device)
    allocator.close_statuses = {11: [sm.CleanupStatus.FAILED]}

    allocator.free(ptr)
    assert allocator.release_pending() == 0
    assert allocator.close_calls == [11, 12]
    assert allocator.free_calls == 0
    assert allocator.cache_stats()["failed"] == 1


def _allocator_with_one_alloc(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FaultInjectingAllocator, int]:
    allocator = _FaultInjectingAllocator()
    monkeypatch.setattr(sm, "_pad_frozen", False)
    monkeypatch.setattr(sm, "cu_memset_d8", lambda *_args: None)
    return allocator, allocator.alloc(_NBYTES, 0)


def test_alloc_layout_and_find_covering(monkeypatch: pytest.MonkeyPatch) -> None:
    allocator, buf = _allocator_with_one_alloc(monkeypatch)
    off = sm.buffer_offset()
    assert buf == 0x100000 + off

    rec, o = allocator.find_covering(buf)
    assert rec is not None and o == 0
    assert rec.block_size == off + sm._round_up(_NBYTES, 16)

    interior, o = allocator.find_covering(buf + 64)
    assert interior is rec and o == 64
    # Inside the physical allocation but outside the data buffer: the front
    # signal pad and the 16-B rounding tail belong to no view.
    assert allocator.find_covering(buf - 8) == (None, 0)
    assert allocator.find_covering(buf + _NBYTES + 8) == (None, 0)

    allocator.free(buf)
    assert allocator.find_covering(buf) == (None, 0)


def test_rendezvous_is_cached_per_ptr_and_comm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator, buf = _allocator_with_one_alloc(monkeypatch)
    device = torch.device("cpu")

    h1 = allocator.rendezvous(buf, 11, 0, 1, device)
    assert allocator.rendezvous(buf, 11, 0, 1, device) is h1
    assert allocator.get_cached_handle(buf, 11) is h1
    assert h1.buffer_ptr - h1.signal_pad_ptr == sm.buffer_offset()
    assert h1.size == _NBYTES

    # A view at an offset gets its own handle but reuses the peer setup.
    h2 = allocator.rendezvous(buf + 64, 11, 0, 1, device)
    assert h2 is not h1 and h2.offset == 64 and h2.size == _NBYTES - 64
    stats = allocator.cache_stats()
    assert stats["peer_alloc_infos"] == 1 and stats["handles"] == 2

    with pytest.raises(ValueError, match="already established with size"):
        allocator.rendezvous(buf, 11, 0, 1, device, size=16)

    allocator.free(buf)
    assert allocator.get_cached_handle(buf, 11) is None
    assert allocator.release_pending() == 1
    assert allocator.close_calls == [11]
    assert allocator.cache_stats()["records"] == 0


def test_rendezvous_rejects_uncovered_ptr(monkeypatch: pytest.MonkeyPatch) -> None:
    allocator, buf = _allocator_with_one_alloc(monkeypatch)
    with pytest.raises(KeyError, match="no symmetric allocation covers"):
        allocator.rendezvous(buf - 8, 11, 0, 1, torch.device("cpu"))
