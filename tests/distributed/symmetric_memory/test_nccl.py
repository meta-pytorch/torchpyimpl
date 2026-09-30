# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the NCCL window backend's failure handling."""

from __future__ import annotations

import weakref
from contextlib import nullcontext

import pytest
import torch

import torchpyimpl.distributed.symmetric_memory._symm_mem as sm
from torchpyimpl.distributed.symmetric_memory.backends import _nccl


class _FakeWindow:
    def __init__(self, fail_close: bool = False, fail_multimem: bool = False) -> None:
        self.closed = False
        self.close_calls = 0
        self.fail_close = fail_close
        self.fail_multimem = fail_multimem

    def get_peer_device_pointer(self, _rank: int) -> int:
        return 0x200000

    def get_lsa_multimem_device_pointer(self, _offset: int) -> int | None:
        if self.fail_multimem:
            raise RuntimeError("injected multimem failure")
        return None

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("injected close failure")
        self.closed = True


class _Group:
    """Stands in for a ProcessGroup that is not in c10d's registry."""


def _record() -> sm.AllocRecord:
    return sm.AllocRecord(
        alloc_base=0x100000,
        block_size=4096,
        buffer_size=1024,
        device=0,
        handle=object(),
    )


def _fail_tensor(*_args: object, **_kwargs: object) -> torch.Tensor:
    raise torch.OutOfMemoryError("injected")


def _cpu_tensor(data: list[int], dtype: torch.dtype, device: object) -> torch.Tensor:
    return torch.zeros(len(data), dtype=dtype)


def _patch_window(
    monkeypatch: pytest.MonkeyPatch, window: _FakeWindow, peer_vote: tuple[bool, bool]
) -> None:
    monkeypatch.setattr(_nccl, "RegisteredWindowHandle", lambda *_args, **_kw: window)
    monkeypatch.setattr(_nccl.nccl_bindings, "Comm", lambda _ptr: object())
    monkeypatch.setattr(_nccl.torch.cuda, "device", lambda _idx: nullcontext())

    def all_gather_object(out: list[object], obj: object, group: object) -> None:
        out[0], out[1] = obj, peer_vote

    monkeypatch.setattr(_nccl.dist, "all_gather_object", all_gather_object)


def _pai(window: _FakeWindow, group: _Group | None = None) -> sm._PeerAllocInfo:
    empty = torch.empty(0, dtype=torch.int64)
    ref = None if group is None else weakref.ref(group)
    nccl_window = _nccl._NcclWindow(window, ref)  # type: ignore[arg-type]
    return sm._PeerAllocInfo(nccl_window, [], [], empty, empty, 0, device=0)


def test_failed_window_close_stays_failed_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _FakeWindow(fail_close=True)
    pai = _pai(window)
    allocator = _nccl.NCCLAllocator()
    monkeypatch.setattr(_nccl, "should_skip_cuda_cleanup", lambda _device: False)

    assert allocator.close_peers(pai) is sm.CleanupStatus.FAILED
    pai.cleanup_failed = True  # as _close_record_peers records it
    assert allocator.close_peers(pai) is sm.CleanupStatus.FAILED
    assert window.close_calls == 1
    assert pai.window is not None


def test_window_of_a_destroyed_group_is_not_deregistered() -> None:
    window = _FakeWindow()
    group = _Group()
    pai = _pai(window, group)
    allocator = _nccl.NCCLAllocator()

    assert not allocator.peers_alive(pai)
    assert allocator.close_peers(pai) is sm.CleanupStatus.CLEANED
    assert window.close_calls == 0


def test_single_rank_rendezvous_oom_closes_unpublished_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _FakeWindow()
    _patch_window(monkeypatch, window, (True, False))
    monkeypatch.setattr(_nccl.torch, "tensor", _fail_tensor)

    with pytest.raises(torch.OutOfMemoryError, match="injected"):
        _nccl.NCCLAllocator().establish_peers(
            _record(), comm_ptr=1, world_size=1, device=torch.device("cuda:0")
        )

    assert window.closed


def test_multirank_rendezvous_oom_closes_window_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This rank fails; the peer's vote is ok, and both ranks close.
    window = _FakeWindow()
    allocator = _nccl.NCCLAllocator()
    _patch_window(monkeypatch, window, (True, False))
    monkeypatch.setattr(_nccl.torch, "tensor", _fail_tensor)

    with pytest.raises(torch.OutOfMemoryError, match="injected"):
        allocator.establish_peers(
            _record(), 1, 2, torch.device("cuda:0"), group=_Group()
        )

    assert window.closed
    assert allocator.cache_stats()["unpublished_failed"] == 0


def test_rendezvous_closes_window_when_a_peer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _FakeWindow()
    _patch_window(monkeypatch, window, (False, False))
    monkeypatch.setattr(_nccl.torch, "tensor", _cpu_tensor)

    with pytest.raises(RuntimeError, match="peer rank failed"):
        _nccl.NCCLAllocator().establish_peers(
            _record(), 1, 2, torch.device("cuda:0"), group=_Group()
        )

    assert window.closed


def test_multirank_rendezvous_needs_the_group() -> None:
    with pytest.raises(ValueError, match="needs the process group"):
        _nccl.NCCLAllocator().establish_peers(
            _record(), comm_ptr=1, world_size=2, device=torch.device("cuda:0")
        )


def test_multimem_query_failure_means_no_multicast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _FakeWindow(fail_multimem=True)
    _patch_window(monkeypatch, window, (True, False))
    monkeypatch.setattr(_nccl.torch, "tensor", _cpu_tensor)

    pai = _nccl.NCCLAllocator().establish_peers(
        _record(), comm_ptr=1, world_size=1, device=torch.device("cuda:0")
    )

    assert pai.multicast_ptr == 0 and pai.multicast_base == 0
    assert not window.closed


def test_rendezvous_rejects_foreign_device() -> None:
    with pytest.raises(ValueError, match="does not own allocation"):
        _nccl.NCCLAllocator().establish_peers(
            _record(), comm_ptr=1, world_size=1, device=torch.device("cuda:1")
        )
