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


# Matches torch's ``default_signal_pad_size``: symm_max_nblocks (32) *
# max_cuda_p2p_domain_size (72) * sizeof(uint32) (4).
DEFAULT_SIGNAL_PAD_BYTES: int = 32 * 72 * 4  # 9216
# torch rounds the signal pad up to 128 (``signal_pad_alignment``) before the
# data buffer; 9216 is already 128-aligned so the default is unaffected.
SIGNAL_PAD_ALIGNMENT: int = 128
_SIGNAL_PAD_ENV: str = "TORCHPYIMPL_SIGNAL_PAD_BYTES"

# The pad size is frozen on the first allocation: ``buffer_offset`` is baked into
# every live allocation's layout and the O(1) lookup, so it must not change once
# any allocation exists.
_pad_frozen: bool = False


def _round_up(n: int, alignment: int) -> int:
    return (n + alignment - 1) & ~(alignment - 1)


def signal_pad_bytes() -> int:
    """The per-segment signal-pad size, in bytes. Configurable via
    ``TORCHPYIMPL_SIGNAL_PAD_BYTES`` / :func:`set_signal_pad_size` before the first
    allocation.

    Validated here, not only in :func:`set_signal_pad_size`: the environment
    variable reaches ``buffer_offset`` without passing through the setter, and a
    non-positive pad would put the data pointer at or before the allocation base
    and leave the signal-pad kernels addressing the data buffer."""
    size = int(os.environ.get(_SIGNAL_PAD_ENV, DEFAULT_SIGNAL_PAD_BYTES))
    if size <= 0:
        raise ValueError(f"{_SIGNAL_PAD_ENV} must be positive, got {size}")
    return size


def signal_pad_slots() -> int:
    """The per-segment signal pad's capacity in ``uint32`` slots — the unit the
    signal-pad protocol addresses (``channel * world_size + rank``)."""
    return signal_pad_bytes() // 4


def buffer_offset() -> int:
    """Constant byte offset from an allocation's base (== signal-pad base) to its
    data buffer: ``round_up(signal_pad_bytes, 128)``. A process-global constant
    once allocations begin (the pad size is frozen), which is what makes
    ``base + buffer_offset`` a sound O(1) recovery of the data pointer."""
    return _round_up(signal_pad_bytes(), SIGNAL_PAD_ALIGNMENT)


def set_signal_pad_size(size: int) -> None:
    """Set the signal-pad size for future allocations. Must be called before the
    first symmetric-memory allocation — the offset is baked in once allocations
    exist, so changing it afterward would desync the layout across ranks."""
    if _pad_frozen:
        raise RuntimeError(
            "set_signal_pad_size must be called before any symmetric-memory "
            "allocation; the size cannot change after allocations exist."
        )
    if size <= 0:
        raise ValueError(f"signal pad size must be positive, got {size}")
    os.environ[_SIGNAL_PAD_ENV] = str(size)


def get_signal_pad_size() -> int:
    """The current per-segment signal-pad size, in bytes."""
    return signal_pad_bytes()


def _device_index(device: torch.device | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    return device.index if device.index is not None else torch.cuda.current_device()


# === records / handles ========================================================


@dataclass
class _PeerAllocInfo:
    """One per (allocation, communicator): owns the single window registration
    and the per-peer pointer arrays. Reused across every view/offset into the
    allocation."""

    window: object  # backend peer state; None once closed or if N/A
    buffers: list[int]  # peer data-buffer ptrs (== signal_pads[i] + buffer_offset)
    signal_pads: list[int]  # peer signal-pad ptrs (== peer allocation/window base)
    buffers_dev: torch.Tensor  # int64 cuda tensor, owns memory
    signal_pads_dev: torch.Tensor
    multicast_ptr: int  # LSA multimem ptr at the data buffer, or 0
    multicast_base: int = 0  # the layout puts the pad at the allocation base
    device: int = 0  # device that owns the mappings and must be synchronized
    cleanup_failed: bool = False


class CleanupStatus(Enum):
    CLEANED = auto()
    DEFERRED = auto()
    FAILED = auto()


@dataclass
class AllocRecord:
    """One physical symmetric allocation, keyed in the allocator by its **data**
    pointer (``alloc_base + buffer_offset``)."""

    alloc_base: int  # signal-pad base == physical / window base
    block_size: int  # buffer_offset + round_up(buffer_size, 16)
    buffer_size: int  # size requested from the allocator (data region)
    device: int  # cuda device index
    handle: object  # backend handle owning the physical memory
    # Allocation order, identical across ranks under SPMD. Deferred releases are
    # drained in this order so the collective window deregisters line up, the
    # way torch's registration_counter does for its own pool enumeration.
    seq: int = 0
    per_comm: dict[int, _PeerAllocInfo] = field(default_factory=dict)
    # Handle cache, subordinate to this record so eviction is automatic: every
    # handle (the base tensor + each slice/alias, keyed on the ptr covering this
    # allocation) lives here and dies when the record is freed. Keeping it in the
    # record — rather than a process-global cache + reverse index — means a freed
    # allocation can never leave a dangling slice handle behind, for any backend.
    handles: dict[tuple[int, int], SymmMemHandle] = field(default_factory=dict)
    # Guards the per_comm get/insert across establish_peers. Not the allocator's
    # _lock: establish_peers allocates through torch and may re-enter our alloc
    # callback, which takes _lock itself.
    reg_lock: threading.Lock = field(default_factory=threading.Lock)
    retired: bool = False
    cleanup_failed: bool = False


@dataclass
class SymmMemHandle:
    """Handle returned by :func:`rendezvous`. Accessor contract:

    * ``buffer_ptrs[i]`` are peer ``i``'s **data-buffer base** pointers (not
      shifted by ``offset``); ``signal_pad_ptrs[i]`` are peer ``i``'s signal-pad
      (== allocation) base pointers.
    * ``buffer_ptr`` is this view's own data address (``buffers[rank] + offset``).
    * ``multicast_ptr`` IS shifted by ``offset``.
    * ``signal_pad_ptrs_tensor`` is the *tensor* behind ``signal_pad_ptrs_dev``
      (which is only its address). Signal-pad kernels take the array as a tensor
      argument — torch's own ``signal_pad_ptrs_dev`` is a Tensor too. It is
      owned by the ``_PeerAllocInfo``, so it stays alive exactly as long as the
      peer mappings it describes.
    """

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
    comm_ptr: int  # communicator pointer or synthetic group id used for rendezvous


# === the shared, backend-agnostic allocator ===================================


class SymmetricMemoryAllocatorBase:
    """Shared allocator machinery. Subclasses implement the backend hooks."""

    requires_nccl_comm: bool = True

    def __init__(self) -> None:
        # keyed on the DATA ptr we hand out; insertion order == allocation order
        # (used for cross-rank-consistent window deregistration at teardown).
        self._records: dict[int, AllocRecord] = {}
        # Allocations the CUDA caching allocator has released but whose windows
        # and physical memory we have not reclaimed yet. See free().
        self._pending: dict[int, AllocRecord] = {}
        # Failed teardown is retained so ownership is explicit and the allocator
        # cannot be switched underneath leaked backend resources.
        self._failed: dict[int, AllocRecord] = {}
        self._unpublished_failed: list[object] = []
        self._seq: int = 0
        self._lock: threading.Lock = threading.Lock()

    # ---- shared alloc: identical layout for every backend --------------------

    def alloc(
        self,
        size: int,
        device: torch.device | int | None = None,
        stream: int | None = None,
    ) -> int:
        """Allocate ``size`` data bytes of symmetric memory (plus the hidden
        front signal pad). Returns the data pointer."""
        idx = _device_index(device)
        # buffer_offset validates the configured pad size and may raise; freeze
        # only once it has, or a bad TORCHPYIMPL_SIGNAL_PAD_BYTES would wedge the
        # process with a frozen pad that set_signal_pad_size can no longer fix.
        off = buffer_offset()
        global _pad_frozen
        _pad_frozen = True
        block_size = off + _round_up(size, 16)

        base, handle = self.alloc_physical(block_size, idx)
        buffer_ptr = base + off
        try:
            # Backend allocations are not guaranteed to be zeroed; initialize
            # the front signal pad before CAS-based signal kernels can observe it.
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
        """Retire the allocation: drop it from lookup and queue it for release.

        Deliberately does no collective and no driver call. The caching allocator
        releases segments in an order that can differ across ranks, so issuing
        collective ``ncclCommWindowDeregister`` calls here could deadlock.
        :meth:`release_pending` does the real work from the ordered teardown path."""
        with self._lock:
            rec = self._records.pop(buffer_ptr, None)
            if rec is None:
                return
            rec.retired = True
            rec.handles.clear()  # subordinate handles die with the record
            self._pending[buffer_ptr] = rec
        logger.info(
            "retire symm segment: base=0x%x buffer=0x%x (release deferred)",
            rec.alloc_base,
            buffer_ptr,
        )

    def release_pending(self, device: torch.device | int | None = None) -> int:
        """Close queued windows and release their memory, in allocation order.

        Must be called from a path the caller controls — never from the
        allocator callback. Sorting on ``seq`` is what makes the collective
        deregisters agree across ranks even though the caching allocator freed
        the segments in its own order."""
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
        """Release peer resources in allocation order.

        Keep the physical memory until the pool releases its segments. All
        ranks use the same allocation order for collective cleanup operations.
        """
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

    # ---- unified O(1) lookup ------------------------------------------------

    def find_covering(self, ptr: int) -> tuple[AllocRecord | None, int]:
        """Return ``(record, offset)`` for the allocation covering ``ptr``.

        Exact-match fast path (``ptr`` is a data-buffer base, the no-split /
        unsliced case); otherwise recover the allocation base via
        ``allocation_base`` and add the constant ``buffer_offset`` — both O(1),
        no linear scan."""
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
        # Covered only within the data buffer [buffer_ptr, buffer_ptr +
        # buffer_size): a ptr in the front signal pad (offset < 0) or past the
        # data region (e.g. the 16-B rounding tail) belongs to no view.
        offset = ptr - data_ptr
        if offset < 0 or offset >= rec.buffer_size:
            return None, 0
        return rec, offset

    # ---- shared rendezvous: never redo cached work --------------------------

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
        """Collective. Build a :class:`SymmMemHandle` for ``buffer_ptr`` over
        ``comm_ptr``.

        Repeated calls with the same ``(buffer_ptr, comm_ptr)`` return the cached
        handle (cached on the covering record, so it is evicted when the
        allocation is freed). The expensive peer setup (``establish_peers``) is
        cached per ``(allocation, comm)``, so a sliced/aliased tensor at a
        different offset reuses it.

        ``comm_ptr`` is the caching identity (a communicator pointer or a
        synthetic group id, stable per group). ``group`` is passed opaquely to
        ``establish_peers`` for backends whose peer exchange is a host
        collective."""
        rec, offset = self.find_covering(buffer_ptr)
        if rec is None:
            raise KeyError(f"no symmetric allocation covers ptr 0x{buffer_ptr:x}")

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
                rec, pai, offset, comm_ptr, rank, world_size, size
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
        )

    def cache_stats(self) -> dict[str, int]:
        """Internal: occupancy counts (used by tests to verify caching)."""
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

    # ---- backend hooks (the ONLY per-backend code) --------------------------

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
        """Release a window registration / peer mappings (backend override).
        Idempotent."""
        return CleanupStatus.CLEANED

    def should_skip_cleanup(self, device: int) -> bool:
        return should_skip_cuda_cleanup(device)


# === backend registry + standalone API ========================================

_BACKENDS: tuple[str, ...] = ("CUDA",)
_active_backend: str = "CUDA"
_allocators: dict[str, SymmetricMemoryAllocatorBase] = {}
_allocators_lock: threading.Lock = threading.Lock()


def _build_allocator(name: str) -> SymmetricMemoryAllocatorBase:
    # Deferred: backends subclass SymmetricMemoryAllocatorBase, so a module-level
    # import would be circular.
    if name == "CUDA":
        from torchpyimpl.distributed.symmetric_memory.backends._cuda_ipc import (
            CUDAIPCAllocator,
        )

        return CUDAIPCAllocator()
    raise ValueError(f"symmetric memory backends are {_BACKENDS}, got {name!r}")


def get_allocator_instance() -> SymmetricMemoryAllocatorBase:
    """The process-wide allocator for the active backend (shared by the MemPool
    allocator callback and the standalone ``alloc``/``rendezvous`` path)."""
    inst = _allocators.get(_active_backend)
    if inst is not None:
        return inst  # hot path: called per alloc/free from the MemPool callback
    with _allocators_lock:
        # Re-check under the lock. Two threads allocating on different devices
        # race here (the caching allocator's device locks do not serialize
        # them), and a second instance would split _records: blocks recorded in
        # the discarded one are invisible to the survivor, so their free() is a
        # silent no-op that leaks the peer state and its physical memory.
        inst = _allocators.get(_active_backend)
        if inst is None:
            inst = _build_allocator(_active_backend)
            _allocators[_active_backend] = inst
    return inst


def get_active_backend() -> str:
    """The active symmetric-memory backend name."""
    return _active_backend


def set_active_backend(name: str) -> None:
    """Switch the active backend. Must be called before the first allocation on
    any backend — the choice is baked into every live allocation's memory."""
    global _active_backend
    if name not in _BACKENDS:
        raise ValueError(f"symmetric memory backends are {_BACKENDS}, got {name!r}")
    if name == _active_backend:
        return
    # _pending counts too: those records are drained by release_pending() on the
    # allocator that owns them, but every drain path resolves the allocator
    # through get_allocator_instance() — i.e. the *active* one. Switching away
    # would strand the old backend's queued releases and physical memory.
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
    """Allocate ``size`` bytes of symmetric memory (active backend); returns the
    data ptr."""
    return get_allocator_instance().alloc(size, device)


def free(ptr: int) -> None:
    """Free the allocation returned by :func:`alloc`.

    Drains immediately: unlike the MemPool callback this runs on the caller's
    thread, so it can do the peer teardown itself."""
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
    """Collective. Build a :class:`SymmMemHandle` for ``buffer_ptr`` (raw-pointer
    entry point; the MemPool path calls the allocator directly). ``group`` is
    required by the CUDA-IPC backend for the IPC-handle exchange."""
    return get_allocator_instance().rendezvous(
        buffer_ptr, comm_ptr, rank, world_size, device, size=size, group=group
    )


def find_covering(ptr: int) -> tuple[AllocRecord | None, int]:
    """Map ``ptr`` to its covering allocation and offset (O(1))."""
    return get_allocator_instance().find_covering(ptr)


def get_cached_handle(buffer_ptr: int, comm_ptr: int) -> SymmMemHandle | None:
    """Return the cached handle for ``(buffer_ptr, comm_ptr)``, or ``None``."""
    return get_allocator_instance().get_cached_handle(buffer_ptr, comm_ptr)


def _cache_stats() -> dict[str, int]:
    """Internal: cache occupancy counts (used by tests)."""
    return get_allocator_instance().cache_stats()


_allocator.set_backend_resolver(get_allocator_instance)
