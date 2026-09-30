# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the cuda-python driver wrappers in ``symmetric_memory._cuda``."""

from __future__ import annotations

import logging

import pytest
import torch
from cuda.bindings import driver
from torchpyimpl.distributed.symmetric_memory._cuda import (
    cu_mem_get_address_range,
    cu_memset_d8,
    cu_stream_write_value32,
)


@pytest.mark.gpu
def test_cu_memset_d8_zeroes_region() -> None:
    device = torch.device("cuda:0")
    t = torch.full((256,), 0xAB, dtype=torch.uint8, device=device)
    torch.cuda.synchronize(device)
    cu_memset_d8(t.data_ptr(), 0, t.numel())
    torch.cuda.synchronize(device)
    assert bool(torch.all(t == 0).item())

    # n == 0 is a no-op: it must not raise or touch memory.
    t.fill_(7)
    torch.cuda.synchronize(device)
    cu_memset_d8(t.data_ptr(), 0, 0)
    torch.cuda.synchronize(device)
    assert bool(torch.all(t == 7).item())

    # n < 0 is a caller bug (underflowed size) and must surface, not be skipped.
    with pytest.raises(ValueError):
        cu_memset_d8(t.data_ptr(), 0, -1)
    torch.cuda.synchronize(device)
    assert bool(torch.all(t == 7).item())


@pytest.mark.gpu
def test_cu_stream_write_value32_writes_one_slot() -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    flags = torch.zeros(4, dtype=torch.int32, device=device)
    stream = torch.cuda.current_stream(device).cuda_stream

    # Writes 4 bytes at the address given, leaving neighbouring slots alone —
    # the caller does its own indexing, so an interior address must work.
    cu_stream_write_value32(flags.data_ptr() + 2 * flags.element_size(), 0xABCD, stream)
    torch.cuda.synchronize(device)
    assert flags.tolist() == [0, 0, 0xABCD, 0]

    # Wider values are masked to the low 32 bits rather than rejected.
    cu_stream_write_value32(flags.data_ptr(), 0x1_0000_0001, stream)
    torch.cuda.synchronize(device)
    assert flags[0].item() == 1


@pytest.mark.gpu
def test_cu_mem_get_address_range_recovers_base() -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=device)  # ensure a live primary context

    nbytes = 1 << 20
    err, dptr = driver.cuMemAlloc(nbytes)
    assert err == driver.CUresult.CUDA_SUCCESS, err
    try:
        base = int(dptr)
        # Any interior pointer resolves to the same allocation base.
        assert cu_mem_get_address_range(base) == base
        assert cu_mem_get_address_range(base + 4096) == base
        assert cu_mem_get_address_range(base + nbytes - 1) == base
    finally:
        (err,) = driver.cuMemFree(dptr)
        assert err == driver.CUresult.CUDA_SUCCESS, err


@pytest.mark.gpu
def test_cu_mem_get_address_range_misses_are_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    torch.cuda.set_device(torch.device("cuda:0"))
    _ = torch.empty(1, device="cuda:0")  # ensure a live primary context

    # Pointers outside any device allocation come back CUDA_ERROR_NOT_FOUND,
    # which is a lookup miss rather than a fault.
    host = torch.zeros(256, dtype=torch.uint8)
    with caplog.at_level(
        logging.DEBUG, logger="torchpyimpl.distributed.symmetric_memory._cuda"
    ):
        assert cu_mem_get_address_range(host.data_ptr()) == 0
        assert cu_mem_get_address_range(0) == 0
        assert cu_mem_get_address_range(0xDEADBEEF000) == 0

    # Each miss leaves a debug trace, so a silent 0 is still diagnosable.
    assert sum("no device allocation covers" in r.message for r in caplog.records) == 3


@pytest.mark.gpu
def test_should_skip_cuda_cleanup_synchronizes_and_allows_cleanup() -> None:
    from torchpyimpl.distributed.symmetric_memory._cuda import should_skip_cuda_cleanup

    dev = torch.cuda.current_device()
    # Real queued work: the guard must drain it rather than return immediately,
    # because ncclMemFree / cuMemUnmap do not wait for kernels.
    torch.cuda._sleep(50_000_000)
    assert should_skip_cuda_cleanup(dev) is False
    assert torch.cuda.current_stream().query(), "guard returned before work drained"


@pytest.mark.gpu
def test_should_skip_cuda_cleanup_declines_on_a_poisoned_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sticky CUDA error must make the guard decline, not raise: raising out of
    teardown replaces the original fault with a confusing one."""
    from torchpyimpl.distributed.symmetric_memory import _cuda

    def _boom() -> None:
        raise RuntimeError("cuCtxSynchronize failed: CUDA_ERROR_ILLEGAL_ADDRESS")

    monkeypatch.setattr(_cuda, "cu_ctx_synchronize", _boom)
    assert _cuda.should_skip_cuda_cleanup(torch.cuda.current_device()) is True


@pytest.mark.gpu
def test_should_skip_cuda_cleanup_does_not_hide_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torchpyimpl.distributed.symmetric_memory import _cuda

    def _boom() -> None:
        raise ValueError("injected invalid cleanup state")

    monkeypatch.setattr(_cuda, "cu_ctx_synchronize", _boom)
    with pytest.raises(ValueError, match="invalid cleanup state"):
        _cuda.should_skip_cuda_cleanup(torch.cuda.current_device())


def test_should_skip_cuda_cleanup_declines_while_finalizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torchpyimpl.distributed.symmetric_memory import _cuda

    monkeypatch.setattr(_cuda.sys, "is_finalizing", lambda: True)
    # Must not touch CUDA at all once the interpreter is going down.
    monkeypatch.setattr(
        _cuda,
        "cu_ctx_synchronize",
        lambda: pytest.fail("touched CUDA while finalizing"),
    )
    assert _cuda.should_skip_cuda_cleanup(0) is True
