# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Manage one symmetric ``torch.cuda.MemPool`` for each device."""

from __future__ import annotations

import logging
import sys
import threading
from enum import Enum

import torch

from torchpyimpl.distributed.symmetric_memory._allocator import (
    alloc,
    free,
    get_backend,
    PoolBackend,
)


logger: logging.Logger = logging.getLogger(__name__)

_lock: threading.Lock = threading.Lock()
_pools: dict[int, torch.cuda.MemPool] = {}  # Store pools by CUDA device index.


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def get_mem_pool(device: torch.device) -> torch.cuda.MemPool:
    """Return the symmetric memory pool for ``device``.

    The function creates the pool when it is first used. ``use_on_oom=False``
    prevents other allocations from using pool memory. This setting keeps the
    allocation state equal on all ranks. ``no_split=True`` gives each segment
    one allocation and one signal pad. It also keeps the tensor data pointer
    equal to the allocation data pointer.
    """
    idx = _device_index(device)
    with _lock:
        if idx not in _pools:
            with torch.cuda.device(idx):
                _pools[idx] = torch.cuda.MemPool.from_py_allocator(
                    alloc,
                    free,
                    use_on_oom=False,
                    no_split=True,
                )
            logger.info("created symmetric MemPool for cuda:%d", idx)
        return _pools[idx]


def _release_peer_resources(allocator: PoolBackend, device: int) -> int | None:
    try:
        return allocator.release_peer_resources(device)
    except Exception:
        logger.warning(
            "peer resource cleanup failed for symmetric MemPool on cuda:%d; "
            "leaving it intact",
            device,
            exc_info=True,
        )
        return None


def _empty_cache(device: int) -> None:
    try:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    except Exception:
        logger.warning(
            "empty_cache failed while freeing symmetric MemPool for cuda:%d",
            device,
            exc_info=True,
        )


def _release_pending(allocator: PoolBackend, device: int) -> int:
    try:
        return allocator.release_pending(device)
    except Exception:
        logger.warning(
            "segment cleanup failed for symmetric MemPool on cuda:%d",
            device,
            exc_info=True,
        )
        return 0


class FreePoolStatus(Enum):
    FREED = "freed"
    NOTHING_TO_FREE = "nothing_to_free"
    IN_USE = "in_use"  # Tensors or outside references keep the pool alive.
    CUDA_ERROR = "cuda_error"  # The backend skips cleanup after a CUDA error.
    INCOMPLETE = "incomplete"  # Cleanup started, and a later call must retry.


def _expected_refcount() -> int:
    control = object()
    holder = {0: control}
    count = sys.getrefcount(control)
    del holder
    return count


def _active_bytes(pool: torch.cuda.MemPool) -> int:
    return sum(segment["active_size"] for segment in pool.snapshot())


def _drained(allocator: PoolBackend, device: int) -> FreePoolStatus:
    if allocator.has_pending_releases(device):
        logger.warning(
            "symmetric-memory releases remain queued on cuda:%d; retry free_pool",
            device,
        )
        return FreePoolStatus.INCOMPLETE
    return FreePoolStatus.FREED


def free_pool(device: torch.device) -> FreePoolStatus:
    """Release the symmetric ``MemPool`` for ``device``.

    First, release peer resources in allocation order. This order is the same
    on all ranks. Then, release the pool and its memory. Keep the pool if it is
    in use, if CUDA fails, or if resource cleanup is incomplete. A later call
    can retry cleanup. The returned status tells which of these occurred.
    """
    idx = _device_index(device)
    with _lock:
        pool = _pools.get(idx)
    allocator = get_backend()
    if pool is None:
        if not allocator.has_pending_releases(idx):
            logger.debug("no symmetric MemPool for cuda:%d; nothing to free", idx)
            return FreePoolStatus.NOTHING_TO_FREE
        if allocator.should_skip_cleanup(idx):
            logger.warning(
                "leaving queued symmetric-memory releases on cuda:%d "
                "after CUDA failure",
                idx,
            )
            return FreePoolStatus.CUDA_ERROR
        released = _release_pending(allocator, idx)
        logger.info(
            "retried symmetric MemPool cleanup on cuda:%d (released %d segment(s))",
            idx,
            released,
        )
        return _drained(allocator, idx)

    if allocator.should_skip_cleanup(idx):
        logger.warning(
            "leaving symmetric MemPool for cuda:%d intact after CUDA failure",
            idx,
        )
        return FreePoolStatus.CUDA_ERROR

    # _pools and the local name above must hold the only references. If
    # another reference exists, deleting the pool does not release its
    # segments.
    if sys.getrefcount(pool) > _expected_refcount() or _active_bytes(pool):
        logger.warning(
            "leaving symmetric MemPool for cuda:%d intact: tensors or other "
            "references still use it",
            idx,
        )
        return FreePoolStatus.IN_USE

    n = _release_peer_resources(allocator, idx)
    if n is None:
        return FreePoolStatus.INCOMPLETE
    if allocator.has_unreleased_resources(idx):
        logger.warning(
            "leaving symmetric MemPool for cuda:%d intact after incomplete "
            "peer resource cleanup",
            idx,
        )
        return FreePoolStatus.INCOMPLETE

    with _lock:
        if _pools.get(idx) is not pool:
            logger.warning(
                "symmetric MemPool for cuda:%d changed during free_pool", idx
            )
            return FreePoolStatus.INCOMPLETE
        del _pools[idx]
    del pool
    # empty_cache releases segments only on the current device. Deleting the
    # pool calls the allocator free callback for all remaining segments. The
    # callback adds each segment to a retryable queue. Drain the queue after all
    # peer resources are released in the same order on all ranks.
    _empty_cache(idx)
    released = _release_pending(allocator, idx)
    logger.info(
        "freed symmetric MemPool for cuda:%d (released %d peer resource(s), "
        "released %d segment(s))",
        idx,
        n,
        released,
    )
    return _drained(allocator, idx)
