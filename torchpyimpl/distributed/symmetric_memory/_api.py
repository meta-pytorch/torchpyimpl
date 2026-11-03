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

from torchpyimpl.distributed.symmetric_memory import _mempool, _symm_mem
from torchpyimpl.distributed.symmetric_memory._comm import get_group_info
from torchpyimpl.distributed.symmetric_memory._symm_mem import signal_pad_bytes


logger: logging.Logger = logging.getLogger(__name__)


def set_backend(name: str) -> None:
    """Select the symmetric-memory backend. Available: ``"CUDA"`` (default),
    CUDA VMM + IPC over fabric handles or POSIX fds.

    Must be called before the first symmetric allocation."""
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
    """Allocate a tensor from the symmetric ``MemPool``.

    The tensor is ordinary; allocating it through the pool routes the underlying
    block through the active symmetric-memory backend, so it can be passed to
    :func:`rendezvous`.
    """
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

    # The allocator callback already recorded this block (keyed on its data
    # pointer) during the allocation above; nothing more to do here.
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
    # Only resolve a real communicator if the active backend dereferences one;
    # otherwise comm_ptr is a synthetic per-group cache key.
    allocator = _symm_mem.get_allocator_instance()
    info = get_group_info(group, device, require_comm=allocator.requires_nccl_comm)
    return allocator.rendezvous(
        int(tensor.data_ptr()),
        info.comm_ptr,
        info.rank,
        info.world_size,
        device,
        group=group,
    )


def free_pool(device: torch.device) -> _mempool.FreePoolStatus:
    """Free the symmetric pool of ``device``. Each rank decides the status
    alone, so the caller must agree on it across ranks. Handles from earlier
    rendezvous calls are not valid after the pool is freed."""
    return _mempool.free_pool(device)


# === handle accessors =========================================================


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
    """The peer's whole data region. ``handle.size`` is the *view* size
    (``buffer_size - offset``), so bounds-checking an offset measured from the
    allocation base against it would count the offset twice."""
    return handle.offset + handle.size


def _handle_device(handle: _symm_mem.SymmMemHandle) -> torch.device:
    """The device the handle's memory lives on. Taken from the peer-pointer
    tensor, which the backend built on the allocation's device — the current
    device may since have drifted, and labelling a view with the wrong one
    silently points at another device's memory."""
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
    """Whether this allocation has a usable LSA/multimem (multicast) pointer.

    A property of the allocation rather than a build-time constant: it depends
    on the backend, the hardware, and — because multicast setup is collective —
    on every rank having succeeded. Guard multimem work on this rather than
    comparing ``multicast_ptr`` yourself.

    The pointer it guards is addressable only through ``multimem.*``
    instructions."""
    return handle.multicast_ptr != 0


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
