# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import GroupName

from torchpyimpl.distributed.symmetric_memory import (
    _mempool,
    _multimem,
    _signal,
    _symm_mem,
)
from torchpyimpl.distributed.symmetric_memory._comm import get_group_info
from torchpyimpl.distributed.symmetric_memory._stream_order import group_stream_guard
from torchpyimpl.distributed.symmetric_memory._symm_mem import (
    barrier_state_offset,
    signal_pad_bytes,
    signal_pad_slots,
)


logger: logging.Logger = logging.getLogger(__name__)


def set_backend(name: str) -> None:
    _symm_mem.set_active_backend(name)


def set_signal_pad_size(size: int) -> None:
    _symm_mem.set_signal_pad_size(size)


def get_signal_pad_size() -> int:
    return _symm_mem.get_signal_pad_size()


def empty(
    *size: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        size = tuple(size[0])

    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise ValueError(f"empty: symmetric memory needs a CUDA device, got {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    pool = _mempool.get_mem_pool(device)
    with torch.cuda.use_mem_pool(pool, device=device):
        t = torch.empty(*size, dtype=dtype, device=device)

    logger.debug(
        "empty: allocated %s %s from symm MemPool at 0x%x",
        tuple(t.shape),
        dtype,
        int(t.data_ptr()),
    )
    return t


def rendezvous(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | str,
) -> _symm_mem.SymmMemHandle:
    if isinstance(group, str):
        group = dist.distributed_c10d._resolve_process_group(GroupName(group))
    device = tensor.device
    allocator = _symm_mem.get_allocator_instance()
    info = get_group_info(group, device, require_comm=allocator.requires_nccl_comm)
    return allocator.rendezvous(
        int(tensor.data_ptr()),
        info.comm_ptr,
        info.rank,
        info.world_size,
        device,
        group=group,
        timeout_ms=info.timeout_ms,
    )


def _resolve_timeout(handle: _symm_mem.SymmMemHandle, timeout_ms: int | None) -> int:
    if timeout_ms is not None:
        if timeout_ms < 0:
            raise ValueError(f"timeout_ms must be >= 0 or None, got {timeout_ms}")
        return timeout_ms
    if handle.group_ref is None or handle.group_ref() is None:
        return 0
    return handle.group_timeout_ms


def free_pool(
    device: torch.device, group: dist.ProcessGroup | None = None
) -> _mempool.FreePoolStatus:
    return _mempool.free_pool(device, group)


def _check_peer(handle: _symm_mem.SymmMemHandle, peer_rank: int, what: str) -> None:
    if not 0 <= peer_rank < handle.world_size:
        raise ValueError(
            f"{what}: peer rank {peer_rank} out of range for world_size "
            f"{handle.world_size}"
        )
    if _symm_mem.get_cached_handle(handle.data_ptr, handle.comm_ptr) is not handle:
        raise RuntimeError(
            f"{what}: the handle is not valid; its allocation or peer mappings "
            "were freed. Call rendezvous again."
        )


def _allocation_bytes(handle: _symm_mem.SymmMemHandle) -> int:
    return handle.offset + handle.size


def _handle_device(handle: _symm_mem.SymmMemHandle) -> torch.device:
    return handle.signal_pad_ptrs_tensor.device


def get_buffer(
    handle: _symm_mem.SymmMemHandle,
    peer_rank: int,
    sizes: tuple[int, ...],
    dtype: torch.dtype,
    storage_offset: int = 0,
) -> torch.Tensor:
    _check_peer(handle, peer_rank, "get_buffer")
    return _peer_tensor(
        handle.buffer_ptrs[peer_rank],
        sizes,
        dtype,
        storage_offset * dtype.itemsize,
        _allocation_bytes(handle),
        _handle_device(handle),
        "get_buffer",
    )


def get_remote_tensor(
    handle: _symm_mem.SymmMemHandle,
    peer_rank: int,
    sizes: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    _check_peer(handle, peer_rank, "get_remote_tensor")
    return _peer_tensor(
        handle.buffer_ptrs[peer_rank],
        sizes,
        dtype,
        handle.offset,
        _allocation_bytes(handle),
        _handle_device(handle),
        "get_remote_tensor",
    )


def get_signal_pad(
    handle: _symm_mem.SymmMemHandle,
    peer_rank: int,
    sizes: tuple[int, ...] | list[int] = (),
    dtype: torch.dtype | None = None,
    storage_offset: int = 0,
) -> torch.Tensor:
    _check_peer(handle, peer_rank, "get_signal_pad")
    pad_size = signal_pad_bytes()
    if dtype is None:
        dtype = torch.uint32
    if not sizes:
        sizes = (pad_size // dtype.itemsize,)
    return _peer_tensor(
        handle.signal_pad_ptrs[peer_rank],
        sizes,
        dtype,
        storage_offset * dtype.itemsize,
        pad_size,
        _handle_device(handle),
        "get_signal_pad",
    )


def has_multicast_support(handle: _symm_mem.SymmMemHandle) -> bool:
    return handle.multicast_ptr != 0


def _check_signal_args(
    handle: _symm_mem.SymmMemHandle, peer_rank: int, channel: int, what: str
) -> None:
    if channel < 0:
        raise ValueError(f"{what}: channel must be non-negative, got {channel}")
    slots = signal_pad_slots()
    needed = (channel + 1) * handle.world_size
    if needed > slots:
        raise ValueError(
            f"{what}: channel {channel} needs {needed} uint32 signal-pad slots "
            f"for world_size {handle.world_size}, but the pad holds {slots} "
            f"({signal_pad_bytes()} bytes). Raise it with set_signal_pad_size() "
            "before the first allocation."
        )
    _check_peer(handle, peer_rank, what)


def barrier(
    handle: _symm_mem.SymmMemHandle, channel: int = 0, timeout_ms: int | None = None
) -> None:
    if handle.world_size <= 1:
        return
    _check_signal_args(handle, handle.rank, channel, "barrier")
    timeout = _resolve_timeout(handle, timeout_ms)
    with torch.cuda.device(_handle_device(handle)), group_stream_guard(handle):
        if handle.multicast_signal_pad_ptr:
            state = barrier_state_offset()
            _multimem.barrier(
                handle.multicast_signal_pad_ptr + state,
                handle.signal_pad_ptr + state,
                handle.world_size,
                channel,
                timeout,
            )
            return
        _signal.barrier(
            handle.signal_pad_ptrs_tensor,
            handle.rank,
            handle.world_size,
            channel,
            timeout,
        )


def put_signal(
    handle: _symm_mem.SymmMemHandle,
    dst_rank: int,
    channel: int = 0,
    timeout_ms: int | None = None,
) -> None:
    _check_signal_args(handle, dst_rank, channel, "put_signal")
    with torch.cuda.device(_handle_device(handle)), group_stream_guard(handle):
        _signal.put_signal(
            handle.signal_pad_ptrs_tensor,
            handle.rank,
            dst_rank,
            handle.world_size,
            channel,
            _resolve_timeout(handle, timeout_ms),
        )


def wait_signal(
    handle: _symm_mem.SymmMemHandle,
    src_rank: int,
    channel: int = 0,
    timeout_ms: int | None = None,
) -> None:
    _check_signal_args(handle, src_rank, channel, "wait_signal")
    with torch.cuda.device(_handle_device(handle)), group_stream_guard(handle):
        _signal.wait_signal(
            handle.signal_pad_ptrs_tensor,
            handle.rank,
            src_rank,
            handle.world_size,
            channel,
            _resolve_timeout(handle, timeout_ms),
        )


def _peer_tensor(
    base_ptr: int,
    sizes: tuple[int, ...] | list[int],
    dtype: torch.dtype,
    offset_bytes: int,
    region_size: int,
    device: torch.device,
    what: str,
) -> torch.Tensor:
    numel = 1
    for s in sizes:
        numel *= s
    tensor_bytes = numel * dtype.itemsize
    req_size = offset_bytes + tensor_bytes
    if offset_bytes < 0 or req_size > region_size:
        raise ValueError(
            f"{what}: requested bytes [{offset_bytes}, {req_size}) are outside "
            f"the allocated {region_size} bytes"
        )
    storage = torch._C._construct_storage_from_data_pointer(
        base_ptr + offset_bytes, device, tensor_bytes
    )
    out = torch.empty(0, dtype=dtype, device=device)
    stride = torch._prims_common.make_contiguous_strides_for(sizes)
    out.set_(storage, 0, sizes, stride)
    return out
