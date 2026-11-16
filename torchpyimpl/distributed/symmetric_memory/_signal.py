# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Signal-pad kernels (CuTeDSL): ``barrier``, ``put_signal``, ``wait_signal``.

Slot for ``src -> dst`` on ``channel``: ``dst_pad[channel * world_size + src]``.
Send is ``CAS(0 -> 1)`` (release, sys); wait is ``CAS(1 -> 0)`` (acquire, sys).
With ``timeout_ms > 0`` a spin traps past its deadline.
"""

from __future__ import annotations

from functools import cache
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda_driver
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm


AddressSpace = cute.AddressSpace

_WARP: int = 32


@cute.jit
def _spin_cas(
    addr: cute.Pointer,
    expect: cutlass.Constexpr[int],
    swap: cutlass.Constexpr[int],
    sem: cutlass.Constexpr[str],
    deadline_ns: Int64,
    bounded: cutlass.Constexpr[bool],
) -> None:
    result = Int32(swap)
    while result != Int32(expect):
        result = cute.arch.atomic_cas(
            ptr=addr.llvm_ptr,
            cmp=Int32(expect),
            val=Int32(swap),
            sem=sem,
            scope="sys",
        )
        if cutlass.const_expr(bounded):
            if result != Int32(expect) and cute.arch.globaltimer() > deadline_ns:
                llvm.intr_trap()


@cute.jit
def _send_signal(
    addr: cute.Pointer, deadline_ns: Int64, bounded: cutlass.Constexpr[bool]
) -> None:
    _spin_cas(addr, 0, 1, "release", deadline_ns, bounded)


@cute.jit
def _await_signal(
    addr: cute.Pointer, deadline_ns: Int64, bounded: cutlass.Constexpr[bool]
) -> None:
    _spin_cas(addr, 1, 0, "acquire", deadline_ns, bounded)


@cute.jit
def _pad_of(pads_addr: Int64, index: Any) -> cute.Pointer:
    pads = cute.make_ptr(Int64, pads_addr, AddressSpace.gmem)
    return cute.make_ptr(
        Int32, cute.arch.load((pads + index).llvm_ptr, Int64), AddressSpace.gmem
    )


@cute.kernel
def _torchpyimpl_put_signal_kernel(
    pads_addr: Int64,
    rank: Int32,
    peer_rank: Int32,
    timeout_ns: Int64,
    world_size: cutlass.Constexpr[int],
    channel: cutlass.Constexpr[int],
    bounded: cutlass.Constexpr[bool],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        deadline = cute.arch.globaltimer() + timeout_ns
        peer_pad = _pad_of(pads_addr, peer_rank)
        _send_signal(peer_pad + (channel * world_size + rank), deadline, bounded)


@cute.kernel
def _torchpyimpl_wait_signal_kernel(
    pads_addr: Int64,
    rank: Int32,
    peer_rank: Int32,
    timeout_ns: Int64,
    world_size: cutlass.Constexpr[int],
    channel: cutlass.Constexpr[int],
    bounded: cutlass.Constexpr[bool],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        deadline = cute.arch.globaltimer() + timeout_ns
        local_pad = _pad_of(pads_addr, rank)
        _await_signal(local_pad + (channel * world_size + peer_rank), deadline, bounded)


@cute.kernel
def _torchpyimpl_signal_barrier_kernel(
    pads_addr: Int64,
    rank: Int32,
    timeout_ns: Int64,
    world_size: cutlass.Constexpr[int],
    channel: cutlass.Constexpr[int],
    bounded: cutlass.Constexpr[bool],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    deadline = cute.arch.globaltimer() + timeout_ns
    local_pad = _pad_of(pads_addr, rank)

    cute.arch.sync_threads()
    if tidx < world_size and tidx != rank:
        peer_pad = _pad_of(pads_addr, tidx)
        _send_signal(peer_pad + (channel * world_size + rank), deadline, bounded)
        _await_signal(local_pad + (channel * world_size + tidx), deadline, bounded)
    cute.arch.sync_threads()


@cute.jit
def _launch_pair(
    pads_addr: Int64,
    rank: Int32,
    peer_rank: Int32,
    timeout_ns: Int64,
    world_size: cutlass.Constexpr[int],
    channel: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
    put: cutlass.Constexpr[bool],
    bounded: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    if cutlass.const_expr(put):
        kernel = _torchpyimpl_put_signal_kernel(
            pads_addr, rank, peer_rank, timeout_ns, world_size, channel, bounded
        )
    else:
        kernel = _torchpyimpl_wait_signal_kernel(
            pads_addr, rank, peer_rank, timeout_ns, world_size, channel, bounded
        )
    kernel.launch(
        grid=[1, 1, 1], block=[threads, 1, 1], cluster=(1, 1, 1), smem=0, stream=stream
    )


@cute.jit
def _launch_barrier(
    pads_addr: Int64,
    rank: Int32,
    timeout_ns: Int64,
    world_size: cutlass.Constexpr[int],
    channel: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
    bounded: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    _torchpyimpl_signal_barrier_kernel(
        pads_addr, rank, timeout_ns, world_size, channel, bounded
    ).launch(
        grid=[1, 1, 1], block=[threads, 1, 1], cluster=(1, 1, 1), smem=0, stream=stream
    )


def _threads(world_size: int) -> int:
    return max(_WARP, -(-world_size // _WARP) * _WARP)


def _stream() -> cuda_driver.CUstream:
    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


def _device() -> int:
    return torch.cuda.current_device()


@cache
def _compiled_pair(
    world_size: int, channel: int, threads: int, put: bool, bounded: bool, device: int
) -> Callable[..., None]:
    return cute.compile(
        _launch_pair,
        Int64(0),
        Int32(0),
        Int32(0),
        Int64(0),
        world_size,
        channel,
        threads,
        put,
        bounded,
        _stream(),
    )


@cache
def _compiled_barrier(
    world_size: int, channel: int, threads: int, bounded: bool, device: int
) -> Callable[..., None]:
    return cute.compile(
        _launch_barrier,
        Int64(0),
        Int32(0),
        Int64(0),
        world_size,
        channel,
        threads,
        bounded,
        _stream(),
    )


def put_signal(
    signal_pad_ptrs: torch.Tensor,
    rank: int,
    dst_rank: int,
    world_size: int,
    channel: int,
    timeout_ms: int = 0,
) -> None:
    _compiled_pair(world_size, channel, _WARP, True, timeout_ms > 0, _device())(
        Int64(signal_pad_ptrs.data_ptr()),
        Int32(rank),
        Int32(dst_rank),
        Int64(max(timeout_ms, 0) * 1_000_000),
        _stream(),
    )


def wait_signal(
    signal_pad_ptrs: torch.Tensor,
    rank: int,
    src_rank: int,
    world_size: int,
    channel: int,
    timeout_ms: int = 0,
) -> None:
    _compiled_pair(world_size, channel, _WARP, False, timeout_ms > 0, _device())(
        Int64(signal_pad_ptrs.data_ptr()),
        Int32(rank),
        Int32(src_rank),
        Int64(max(timeout_ms, 0) * 1_000_000),
        _stream(),
    )


def barrier(
    signal_pad_ptrs: torch.Tensor,
    rank: int,
    world_size: int,
    channel: int,
    timeout_ms: int = 0,
) -> None:
    _compiled_barrier(
        world_size, channel, _threads(world_size), timeout_ms > 0, _device()
    )(
        Int64(signal_pad_ptrs.data_ptr()),
        Int32(rank),
        Int64(max(timeout_ms, 0) * 1_000_000),
        _stream(),
    )
