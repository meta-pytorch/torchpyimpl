# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import re
import time
import weakref
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

import torchpyimpl.distributed.symmetric_memory._symm_mem as sm
from torchpyimpl.distributed.symmetric_memory import _api as api, _signal


_SYM = re.compile(r"kernel_cutlass__[A-Za-z0-9_]+")


def _dummy_handle(world_size: int) -> sm.SymmMemHandle:
    return sm.SymmMemHandle(
        buffer_ptr=0,
        signal_pad_ptr=0,
        buffer_ptrs=[0] * world_size,
        signal_pad_ptrs=[0] * world_size,
        buffer_ptrs_dev=0,
        signal_pad_ptrs_dev=0,
        signal_pad_ptrs_tensor=torch.zeros(world_size, dtype=torch.int64),
        multicast_ptr=0,
        multicast_signal_pad_ptr=0,
        size=0,
        offset=0,
        rank=0,
        world_size=world_size,
        comm_ptr=0,
    )


def test_signal_rejects_out_of_range_peer() -> None:
    hdl = _dummy_handle(8)
    with pytest.raises(ValueError):
        hdl.put_signal(8)
    with pytest.raises(ValueError):
        hdl.wait_signal(-1)


def test_signal_rejects_channel_past_the_pad() -> None:
    hdl = _dummy_handle(8)
    with pytest.raises(ValueError):
        hdl.put_signal(0, channel=-1)
    with pytest.raises(ValueError):
        hdl.barrier(channel=sm.signal_pad_slots() // 8)


def test_barrier_is_a_noop_for_one_rank() -> None:
    _dummy_handle(1).barrier()


def test_timeout_none_uses_the_group_timeout_and_zero_is_unbounded() -> None:
    class _Group:
        pass

    group = _Group()
    hdl = _dummy_handle(2)
    hdl.group_ref = weakref.ref(group)  # type: ignore[assignment]
    hdl.group_timeout_ms = 5000
    assert api._resolve_timeout(hdl, None) == 5000
    assert api._resolve_timeout(hdl, 0) == 0
    assert api._resolve_timeout(hdl, 7) == 7
    with pytest.raises(ValueError):
        api._resolve_timeout(hdl, -1)
    del group
    gc.collect()
    assert api._resolve_timeout(hdl, None) == 0
    assert api._resolve_timeout(_dummy_handle(2), None) == 0


def _only_symbol(compiled: Any) -> str:
    symbols = sorted(set(_SYM.findall(str(compiled.ir_module))))
    assert len(symbols) == 1, f"expected exactly one kernel, got {symbols}"
    return symbols[0]


@pytest.mark.gpu
def test_signal_barrier_kernel_is_named() -> None:
    torch.cuda.set_device(0)
    symbol = _only_symbol(_signal._compiled_barrier(2, 0, 32, True, 0))
    assert "torchpyimpl_signal_barrier_kernel" in symbol, symbol


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("put", "expected"),
    [
        (True, "torchpyimpl_put_signal_kernel"),
        (False, "torchpyimpl_wait_signal_kernel"),
    ],
)
def test_signal_pair_kernels_are_named(put: bool, expected: str) -> None:
    torch.cuda.set_device(0)
    symbol = _only_symbol(_signal._compiled_pair(2, 0, 32, put, True, 0))
    assert expected in symbol, symbol


def _never_signalled_worker(rank: int) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    pad = torch.zeros(256, dtype=torch.int32, device=device)
    pads = torch.tensor([pad.data_ptr()], dtype=torch.int64, device=device)
    torch.cuda.synchronize(device)

    _signal.wait_signal(pads, 0, 0, world_size=1, channel=0, timeout_ms=1000)
    torch.cuda.synchronize(device)


@pytest.mark.gpu
def test_signal_timeout_faults_instead_of_hanging() -> None:
    start = time.monotonic()
    with pytest.raises(mp.ProcessRaisedException):
        mp.spawn(_never_signalled_worker, nprocs=1, join=True)
    elapsed = time.monotonic() - start
    assert elapsed < 60, f"spin was not bounded (took {elapsed:.0f}s)"
