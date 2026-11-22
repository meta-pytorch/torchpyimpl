# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import struct
from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda_driver
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm


AddressSpace = cute.AddressSpace

_VEC: int = 4
_THREADS: int = 128
_INT32_MAX: int = 2**31 - 1

_ST_V4_PTX: str = "multimem.st.relaxed.sys.global.v4.f32 [$0], {$1, $2, $3, $4};"

_RED_ADD1_PTX: str = "multimem.red.release.sys.global.add.u32 [$0], 1;"


@cute.jit
def _multimem_st_v4(ptr: cute.Pointer, bits: Int32) -> None:
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value()] + [bits.ir_value()] * 4,
        _ST_V4_PTX,
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _multimem_red_add1(ptr: cute.Pointer) -> None:
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value()],
        _RED_ADD1_PTX,
        "l",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.kernel
def _torchpyimpl_multimem_barrier_kernel(
    mc_state_addr: Int64,
    local_state_addr: Int64,
    channel: Int32,
    timeout_ns: Int64,
    world_size: cutlass.Constexpr[int],
    bounded: cutlass.Constexpr[bool],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        deadline = cute.arch.globaltimer() + timeout_ns

        mc_flag = cute.make_ptr(Int32, mc_state_addr, AddressSpace.gmem) + channel
        local_flag = cute.make_ptr(Int32, local_state_addr, AddressSpace.gmem) + channel

        _multimem_red_add1(mc_flag)
        seen = Int32(0)
        while seen < Int32(world_size):
            seen = cute.arch.load(
                local_flag.llvm_ptr, Int32, sem="acquire", scope="sys"
            )
            if cutlass.const_expr(bounded):
                if seen < Int32(world_size) and cute.arch.globaltimer() > deadline:
                    llvm.intr_trap()

        cute.arch.atomic_add(
            local_flag.llvm_ptr, Int32(-world_size), sem="relaxed", scope="sys"
        )


@cute.jit
def _launch_barrier(
    mc_state_addr: Int64,
    local_state_addr: Int64,
    channel: Int32,
    timeout_ns: Int64,
    world_size: cutlass.Constexpr[int],
    bounded: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    _torchpyimpl_multimem_barrier_kernel(
        mc_state_addr, local_state_addr, channel, timeout_ns, world_size, bounded
    ).launch(grid=[1, 1, 1], block=[1, 1, 1], cluster=(1, 1, 1), smem=0, stream=stream)


@cute.kernel
def _torchpyimpl_multimem_fill_kernel(
    mc_addr: Int64, bits: Int32, n_vec: Int32
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    if i < n_vec:
        _multimem_st_v4(
            cute.make_ptr(Int32, mc_addr + Int64(i) * 16, AddressSpace.gmem), bits
        )


@cute.jit
def _launch_fill(
    mc_addr: Int64,
    bits: Int32,
    n_vec: Int32,
    blocks: Int32,
    threads: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
) -> None:
    _torchpyimpl_multimem_fill_kernel(mc_addr, bits, n_vec).launch(
        grid=[blocks, 1, 1],
        block=[threads, 1, 1],
        cluster=(1, 1, 1),
        smem=0,
        stream=stream,
    )


def _stream() -> cuda_driver.CUstream:
    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


@cache
def _compiled_fill(threads: int, device: int) -> Any:
    return cute.compile(
        _launch_fill, Int64(0), Int32(0), Int32(0), Int32(0), threads, _stream()
    )


@cache
def _compiled_barrier(world_size: int, bounded: bool, device: int) -> Any:
    return cute.compile(
        _launch_barrier,
        Int64(0),
        Int64(0),
        Int32(0),
        Int64(0),
        world_size,
        bounded,
        _stream(),
    )


def barrier(
    multicast_state_ptr: int,
    state_ptr: int,
    world_size: int,
    channel: int,
    timeout_ms: int = 0,
) -> None:
    if multicast_state_ptr == 0:
        raise ValueError("multimem barrier on a null multicast barrier state")
    _compiled_barrier(world_size, timeout_ms > 0, torch.cuda.current_device())(
        Int64(multicast_state_ptr),
        Int64(state_ptr),
        Int32(channel),
        Int64(max(timeout_ms, 0) * 1_000_000),
        _stream(),
    )


def fill(multicast_ptr: int, numel: int, value: float) -> None:
    if numel % _VEC:
        raise ValueError(f"multimem fill needs numel % {_VEC} == 0, got {numel}")
    if multicast_ptr == 0:
        raise ValueError("multimem fill on a null multicast pointer")
    if multicast_ptr % 16:
        raise ValueError(
            f"multimem fill needs a 16-byte aligned pointer, got 0x{multicast_ptr:x}"
        )
    if numel == 0:
        return
    n_vec = numel // _VEC
    if n_vec > _INT32_MAX:
        raise ValueError(f"multimem fill supports at most {_INT32_MAX * _VEC} elements")
    bits = struct.unpack("<i", struct.pack("<f", value))[0]
    _compiled_fill(_THREADS, torch.cuda.current_device())(
        Int64(multicast_ptr),
        Int32(bits),
        Int32(n_vec),
        Int32(-(-n_vec // _THREADS)),
        _stream(),
    )
