# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-agnostic symmetric-memory allocator.

Each allocation is ``[signal pad | data buffer]``; ``alloc`` returns the data
pointer, ``buffer_offset`` bytes past the base. Backends implement
``alloc_physical``, ``free_physical``, ``allocation_base``, ``establish_peers``
and ``close_peers``.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from enum import auto, Enum

import torch

from torchpyimpl.distributed.symmetric_memory import _allocator
from torchpyimpl.distributed.symmetric_memory._cuda import (
    cu_memset_d8,
    should_skip_cuda_cleanup,
)


logger: logging.Logger = logging.getLogger(__name__)


DEFAULT_SIGNAL_PAD_BYTES: int = 32 * 72 * 4
SIGNAL_PAD_ALIGNMENT: int = 128
_SIGNAL_PAD_ENV: str = "TORCHPYIMPL_SIGNAL_PAD_BYTES"

_signal_pad_size: int | None = None
_pad_frozen: bool = False


def _round_up(n: int, alignment: int) -> int:
    return (n + alignment - 1) & ~(alignment - 1)


def signal_pad_bytes() -> int:
    if _signal_pad_size is not None:
        return _signal_pad_size
    size = int(os.environ.get(_SIGNAL_PAD_ENV, DEFAULT_SIGNAL_PAD_BYTES))
    if size <= 0:
        raise ValueError(f"{_SIGNAL_PAD_ENV} must be positive, got {size}")
    return size


def signal_pad_slots() -> int:
    return signal_pad_bytes() // 4


def buffer_offset() -> int:
    return 3 * _round_up((signal_pad_bytes() + 1) // 2, SIGNAL_PAD_ALIGNMENT)


def barrier_state_offset() -> int:
    return buffer_offset() // 3 * 2


def set_signal_pad_size(size: int) -> None:
    global _signal_pad_size
    if _pad_frozen:
        raise RuntimeError(
            "set_signal_pad_size must be called before any symmetric-memory "
            "allocation; the size cannot change after allocations exist."
        )
    if size <= 0:
        raise ValueError(f"signal pad size must be positive, got {size}")
    _signal_pad_size = size


def get_signal_pad_size() -> int:
    return signal_pad_bytes()


def _device_index(device: torch.device | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    return device.index if device.index is not None else torch.cuda.current_device()


@dataclass
class _PeerAllocInfo:
    window: object
    buffers: list[int]
    signal_pads: list[int]
    buffers_dev: torch.Tensor
    signal_pads_dev: torch.Tensor
    multicast_ptr: int
    multicast_base: int = 0
    device: int = 0
    cleanup_failed: bool = False


class CleanupStatus(Enum):
    CLEANED = auto()
    DEFERRED = auto()
    FAILED = auto()


@dataclass
class AllocRecord:
    alloc_base: int
    block_size: int
    buffer_size: int
    device: int
    handle: object
    seq: int = 0
    per_comm: dict[int, _PeerAllocInfo] = field(default_factory=dict)
    handles: dict[tuple[int, int], SymmMemHandle] = field(default_factory=dict)
    reg_lock: threading.Lock = field(default_factory=threading.Lock)
    retired: bool = False
    cleanup_failed: bool = False


@dataclass
class SymmMemHandle:
    buffer_ptr: int
    signal_pad_ptr: int
    buffer_ptrs: list[int]
    signal_pad_ptrs: list[int]
    buffer_ptrs_dev: int
    signal_pad_ptrs_dev: int
    signal_pad_ptrs_tensor: torch.Tensor
    multicast_ptr: int
    multicast_signal_pad_ptr: int
    size: int
    offset: int
    rank: int
    world_size: int
    comm_ptr: int
    data_ptr: int = 0

    def barrier(self, channel: int = 0, timeout_ms: int | None = None) -> None:
        from torchpyimpl.distributed.symmetric_memory import _api

        _api.barrier(self, channel, timeout_ms)

    def put_signal(
        self, dst_rank: int, channel: int = 0, timeout_ms: int | None = None
    ) -> None:
        from torchpyimpl.distributed.symmetric_memory import _api

        _api.put_signal(self, dst_rank, channel, timeout_ms)

    def wait_signal(
        self, src_rank: int, channel: int = 0, timeout_ms: int | None = None
    ) -> None:
        from torchpyimpl.distributed.symmetric_memory import _api

        _api.wait_signal(self, src_rank, channel, timeout_ms)


class SymmetricMemoryAllocatorBase:
    requires_nccl_comm: bool = True

    def __init__(self) -> None:
        self._records: dict[int, AllocRecord] = {}
        self._pending: dict[int, AllocRecord] = {}
        self._failed: dict[int, AllocRecord] = {}
        self._unpublished_failed: list[object] = []
        self._seq: int = 0
        self._lock: threading.Lock = threading.Lock()

    def alloc(
        self,
        size: int,
        device: torch.device | int | None = None,
        stream: int | None = None,
    ) -> int:
        idx = _device_index(device)
        global _signal_pad_size, _pad_frozen
        _signal_pad_size = signal_pad_bytes()
        _pad_frozen = True
        off = buffer_offset()
        block_size = off + _round_up(size, 16)

        base, handle = self.alloc_physical(block_size, idx)
        buffer_ptr = base + off
        try:
            with torch.cuda.device(idx):
                self.initialize_physical(handle, base, off)
            with self._lock:
                self._records[buffer_ptr] = AllocRecord(
                    alloc_base=base,
                    block_size=block_size,
                    buffer_size=size,
                    device=idx,
                    handle=handle,
                    seq=self._seq,
                )
                self._seq += 1
        except BaseException as e:
            status = CleanupStatus.FAILED
            try:
                status = self.free_physical(handle)
            except Exception:
                logger.exception(
                    "failed to roll back unpublished symm allocation base=0x%x",
                    base,
                )
            if status is not CleanupStatus.CLEANED:
                logger.warning(
                    "could not roll back unpublished symm allocation base=0x%x (%s)",
                    base,
                    status.name.lower(),
                )
                rec = AllocRecord(
                    alloc_base=base,
                    block_size=block_size,
                    buffer_size=size,
                    device=idx,
                    handle=handle,
                    seq=self._seq,
                    retired=True,
                    cleanup_failed=status is CleanupStatus.FAILED,
                )
                with self._lock:
                    if status is CleanupStatus.DEFERRED:
                        self._pending[buffer_ptr] = rec
                    else:
                        self._failed[buffer_ptr] = rec
                    self._seq += 1
            raise e
        logger.info(
            "alloc symm segment: base=0x%x buffer=0x%x size=%d (+pad %d = %d)",
            base,
            buffer_ptr,
            size,
            off,
            block_size,
        )
        return buffer_ptr

    def free(
        self,
        buffer_ptr: int,
        size: int | None = None,
        device: int | None = None,
        stream: int | None = None,
    ) -> None:
        with self._lock:
            rec = self._records.pop(buffer_ptr, None)
            if rec is None:
                return
            rec.retired = True
            rec.handles.clear()
            self._pending[buffer_ptr] = rec
        logger.info(
            "retire symm segment: base=0x%x buffer=0x%x (release deferred)",
            rec.alloc_base,
            buffer_ptr,
        )

    def release_pending(self, device: torch.device | int | None = None) -> int:
        idx = None if device is None else _device_index(device)
        with self._lock:
            items = [
                (bp, r)
                for bp, r in self._pending.items()
                if idx is None or r.device == idx
            ]
            items.sort(key=lambda kv: kv[1].seq)
            for bp, _rec in items:
                del self._pending[bp]
        released = 0
        for bp, rec in items:
            if rec.cleanup_failed:
                physical_status = CleanupStatus.FAILED
            else:
                peer_status, _ = self._close_record_peers(rec)
                physical_status = peer_status
            if physical_status is CleanupStatus.CLEANED:
                physical_status = self._free_record_physical(rec, bp)

            if physical_status is CleanupStatus.CLEANED:
                released += 1
                logger.info(
                    "release symm segment: base=0x%x buffer=0x%x",
                    rec.alloc_base,
                    bp,
                )
            elif physical_status is CleanupStatus.DEFERRED:
                with self._lock:
                    self._pending[bp] = rec
                logger.warning(
                    "deferred symm segment release: base=0x%x buffer=0x%x",
                    rec.alloc_base,
                    bp,
                )
            else:
                rec.cleanup_failed = True
                with self._lock:
                    self._failed[bp] = rec
                logger.error(
                    "failed to release symm segment base=0x%x buffer=0x%x; "
                    "retaining ownership as leaked",
                    rec.alloc_base,
                    bp,
                )
        return released

    def _close_record_peers(self, rec: AllocRecord) -> tuple[CleanupStatus, int]:
        overall = CleanupStatus.CLEANED
        cleaned = 0
        with rec.reg_lock:
            for comm_ptr, pai in list(rec.per_comm.items()):
                try:
                    status = self.close_peers(pai)
                except Exception:
                    logger.exception(
                        "failed to close symm window base=0x%x comm=0x%x",
                        rec.alloc_base,
                        comm_ptr,
                    )
                    status = CleanupStatus.FAILED
                if status is CleanupStatus.CLEANED:
                    rec.per_comm.pop(comm_ptr, None)
                    for key in [k for k in rec.handles if k[1] == comm_ptr]:
                        rec.handles.pop(key, None)
                    cleaned += 1
                if status is CleanupStatus.FAILED:
                    pai.cleanup_failed = True
                    overall = CleanupStatus.FAILED
                elif (
                    status is CleanupStatus.DEFERRED
                    and overall is CleanupStatus.CLEANED
                ):
                    overall = CleanupStatus.DEFERRED
        return overall, cleaned

    def _free_record_physical(self, rec: AllocRecord, buffer_ptr: int) -> CleanupStatus:
        try:
            return self.free_physical(rec.handle)
        except Exception:
            logger.exception(
                "failed to free symm segment base=0x%x buffer=0x%x",
                rec.alloc_base,
                buffer_ptr,
            )
            return CleanupStatus.FAILED

    def release_peer_resources(self, device: torch.device | int | None = None) -> int:
        idx = _device_index(device)
        with self._lock:
            recs = [(bp, r) for bp, r in self._records.items() if r.device == idx]
        n = 0
        for _buffer_ptr, rec in recs:
            status, cleaned = self._close_record_peers(rec)
            n += cleaned
            if status is CleanupStatus.FAILED:
                rec.cleanup_failed = True
        return n

    def has_unreleased_resources(self, device: torch.device | int) -> bool:
        idx = _device_index(device)
        with self._lock:
            return any(
                rec.device == idx
                and any(
                    pai.window is not None and not pai.cleanup_failed
                    for pai in rec.per_comm.values()
                )
                for rec in self._records.values()
            )

    def has_pending_releases(self, device: torch.device | int) -> bool:
        idx = _device_index(device)
        with self._lock:
            return any(rec.device == idx for rec in self._pending.values())

    def retain_unpublished_failure(self, resource: object) -> None:
        with self._lock:
            self._unpublished_failed.append(resource)

    def find_covering(self, ptr: int) -> tuple[AllocRecord | None, int]:
        with self._lock:
            rec = self._records.get(ptr)
        if rec is not None:
            return rec, 0
        base = self.allocation_base(ptr)
        if base == 0:
            return None, 0
        data_ptr = base + buffer_offset()
        with self._lock:
            rec = self._records.get(data_ptr)
        if rec is None:
            return None, 0
        offset = ptr - data_ptr
        if offset < 0 or offset >= rec.buffer_size:
            return None, 0
        return rec, offset

    def rendezvous(
        self,
        buffer_ptr: int,
        comm_ptr: int,
        rank: int,
        world_size: int,
        device: torch.device,
        size: int | None = None,
        group: object | None = None,
    ) -> SymmMemHandle:
        rec, offset = self.find_covering(buffer_ptr)
        if rec is None:
            raise KeyError(f"no symmetric allocation covers ptr 0x{buffer_ptr:x}")
        available = rec.buffer_size - offset
        if size is not None and not 0 < size <= available:
            raise ValueError(
                f"rendezvous: size {size} is not in (0, {available}] for "
                f"0x{buffer_ptr:x}"
            )

        with rec.reg_lock:
            stale = rec.per_comm.get(comm_ptr)
            if stale is not None and not self.peers_alive(stale):
                del rec.per_comm[comm_ptr]
                with self._lock:
                    for k in [k for k in rec.handles if k[1] == comm_ptr]:
                        del rec.handles[k]

        key = (buffer_ptr, comm_ptr)
        with self._lock:
            cached = rec.handles.get(key)
            retired = rec.retired
        if retired:
            raise RuntimeError("symmetric allocation was freed during rendezvous")
        if cached is not None:
            want = rec.buffer_size - offset if size is None else size
            if cached.size != want:
                raise ValueError(
                    f"rendezvous: 0x{buffer_ptr:x} on comm 0x{comm_ptr:x} is "
                    f"already established with size {cached.size}, but this call "
                    f"asks for {want}. The cache is keyed on (ptr, comm), so "
                    "returning the existing handle would hand back the wrong "
                    "size; free and re-rendezvous, or use a distinct view."
                )
            return cached

        with rec.reg_lock:
            pai = rec.per_comm.get(comm_ptr)
            if pai is None:
                pai = self.establish_peers(rec, comm_ptr, world_size, device, group)
                rec.per_comm[comm_ptr] = pai
            handle = self._make_handle(
                rec, pai, buffer_ptr, offset, comm_ptr, rank, world_size, size
            )
            with self._lock:
                if rec.retired:
                    raise RuntimeError(
                        "symmetric allocation was freed during rendezvous"
                    )
                rec.handles[key] = handle
        return handle

    def get_cached_handle(self, buffer_ptr: int, comm_ptr: int) -> SymmMemHandle | None:
        rec, _ = self.find_covering(buffer_ptr)
        if rec is None:
            return None
        with self._lock:
            return rec.handles.get((buffer_ptr, comm_ptr))

    def _make_handle(
        self,
        rec: AllocRecord,
        pai: _PeerAllocInfo,
        data_ptr: int,
        offset: int,
        comm_ptr: int,
        rank: int,
        world_size: int,
        size: int | None,
    ) -> SymmMemHandle:
        multicast_ptr = pai.multicast_ptr + offset if pai.multicast_ptr != 0 else 0
        return SymmMemHandle(
            buffer_ptr=pai.buffers[rank] + offset,
            signal_pad_ptr=pai.signal_pads[rank],
            buffer_ptrs=pai.buffers,
            signal_pad_ptrs=pai.signal_pads,
            buffer_ptrs_dev=int(pai.buffers_dev.data_ptr()),
            signal_pad_ptrs_dev=int(pai.signal_pads_dev.data_ptr()),
            signal_pad_ptrs_tensor=pai.signal_pads_dev,
            multicast_ptr=multicast_ptr,
            multicast_signal_pad_ptr=pai.multicast_base,
            size=size if size is not None else rec.buffer_size - offset,
            offset=offset,
            rank=rank,
            world_size=world_size,
            comm_ptr=comm_ptr,
            data_ptr=data_ptr,
        )

    def cache_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "records": len(self._records),
                "peer_alloc_infos": sum(
                    len(r.per_comm) for r in self._records.values()
                ),
                "handles": sum(len(r.handles) for r in self._records.values()),
                "pending": len(self._pending),
                "failed": len(self._failed),
                "unpublished_failed": len(self._unpublished_failed),
            }

    def alloc_physical(self, block_size: int, device: int) -> tuple[int, object]:
        raise NotImplementedError

    def initialize_physical(self, handle: object, base: int, size: int) -> None:
        cu_memset_d8(base, 0, size)

    def free_physical(self, handle: object) -> CleanupStatus:
        raise NotImplementedError

    def allocation_base(self, ptr: int) -> int:
        raise NotImplementedError

    def establish_peers(
        self,
        rec: AllocRecord,
        comm_ptr: int,
        world_size: int,
        device: torch.device,
        group: object | None = None,
    ) -> _PeerAllocInfo:
        raise NotImplementedError

    def close_peers(self, pai: _PeerAllocInfo) -> CleanupStatus:
        return CleanupStatus.CLEANED

    def peers_alive(self, pai: _PeerAllocInfo) -> bool:
        return True

    def should_skip_cleanup(self, device: int) -> bool:
        return should_skip_cuda_cleanup(device)


_BACKENDS: tuple[str, ...] = ("CUDA", "NCCL")
_active_backend: str = "CUDA"
_allocators: dict[str, SymmetricMemoryAllocatorBase] = {}
_allocators_lock: threading.Lock = threading.Lock()


def _build_allocator(name: str) -> SymmetricMemoryAllocatorBase:
    if name == "CUDA":
        from torchpyimpl.distributed.symmetric_memory.backends._cuda_ipc import (
            CUDAIPCAllocator,
        )

        return CUDAIPCAllocator()
    if name == "NCCL":
        from torchpyimpl.distributed.symmetric_memory.backends._nccl import (
            NCCLAllocator,
        )

        return NCCLAllocator()
    raise ValueError(f"symmetric memory backends are {_BACKENDS}, got {name!r}")


def get_allocator_instance() -> SymmetricMemoryAllocatorBase:
    inst = _allocators.get(_active_backend)
    if inst is not None:
        return inst
    with _allocators_lock:
        inst = _allocators.get(_active_backend)
        if inst is None:
            inst = _build_allocator(_active_backend)
            _allocators[_active_backend] = inst
    return inst


def get_active_backend() -> str:
    return _active_backend


def set_active_backend(name: str) -> None:
    global _active_backend
    if name not in _BACKENDS:
        raise ValueError(f"symmetric memory backends are {_BACKENDS}, got {name!r}")
    if name == _active_backend:
        return
    live = [
        n
        for n, a in _allocators.items()
        if a._records or a._pending or a._failed or a._unpublished_failed
    ]
    if live:
        raise RuntimeError(
            f"cannot switch symmetric-memory backend to {name!r}: allocations "
            f"already exist on {live} (live, or freed and awaiting release). "
            "Call set_backend before the first allocation."
        )
    _active_backend = name


def alloc(size: int, device: torch.device | None = None) -> int:
    return get_allocator_instance().alloc(size, device)


def free(ptr: int) -> None:
    allocator = get_allocator_instance()
    allocator.free(ptr)
    allocator.release_pending()


def rendezvous(
    buffer_ptr: int,
    size: int,
    comm_ptr: int,
    rank: int,
    world_size: int,
    device: torch.device,
    group: object | None = None,
) -> SymmMemHandle:
    return get_allocator_instance().rendezvous(
        buffer_ptr, comm_ptr, rank, world_size, device, size=size, group=group
    )


def find_covering(ptr: int) -> tuple[AllocRecord | None, int]:
    return get_allocator_instance().find_covering(ptr)


def get_cached_handle(buffer_ptr: int, comm_ptr: int) -> SymmMemHandle | None:
    return get_allocator_instance().get_cached_handle(buffer_ptr, comm_ptr)


def _cache_stats() -> dict[str, int]:
    return get_allocator_instance().cache_stats()


_allocator.set_backend_resolver(get_allocator_instance)
