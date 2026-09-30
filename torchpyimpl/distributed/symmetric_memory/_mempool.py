# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The symmetric-memory ``MemPool``, one per device, over the active backend.

Mirrors ``torch.distributed._symmetric_memory.get_mem_pool``:
``torch.cuda.MemPool(allocator, use_on_oom=False, no_split=True)``.

Segment lookup and window lifetime live in the backend (:class:`PoolBackend`),
not here: every block the caching allocator requests is recorded by the
allocator's alloc callback (keyed on the data pointer), and freed by its free
callback. :func:`free_pool` deregisters the device's windows in allocation
order first (collective-safe), then drops the pool so the caching allocator
frees the segments.
"""

from __future__ import annotations

import logging
import threading

import torch
from torchpyimpl.distributed.symmetric_memory._allocator import (
    _get_allocator,
    get_backend,
    PoolBackend,
)

logger: logging.Logger = logging.getLogger(__name__)

_lock: threading.Lock = threading.Lock()
_pools: dict[int, torch.cuda.MemPool] = {}  # device index -> pool


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def get_mem_pool(device: torch.device) -> torch.cuda.MemPool:
    """The symmetric-memory ``MemPool`` for ``device`` (created on first use).

    Preset options mirror torch exactly:

    * ``use_on_oom=False`` — never lend the pool's space to non-symmetric
      allocations; doing so would desync the allocation state across ranks.
    * ``no_split=True`` — one allocation per segment, because a segment carries
      exactly one signal pad; two tensors sharing a segment would share a signal
      pad and concurrent kernels would race on it. It also keeps a tensor's
      ``data_ptr`` equal to its allocation's data pointer, so ``rendezvous`` hits
      the allocator's exact-match fast path.
    """
    idx = _device_index(device)
    with _lock:
        if idx not in _pools:
            # _get_allocator is unsynchronized; this lock is what serializes the
            # one-time build of its ctypes callbacks. Keep the call inside it.
            _pools[idx] = torch.cuda.MemPool(
                _get_allocator(),
                use_on_oom=False,
                no_split=True,
            )
            logger.info("created symmetric MemPool for cuda:%d", idx)
        return _pools[idx]


def _deregister_windows(allocator: PoolBackend, device: int) -> int | None:
    try:
        return allocator.deregister_windows(device)
    except Exception:
        logger.warning(
            "window cleanup failed for symmetric MemPool on cuda:%d; leaving it intact",
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


def free_pool(device: torch.device) -> None:
    """Tear down the symmetric ``MemPool`` for ``device``.

    Deregisters every symmetric window over the device's allocations in
    allocation order first — so the collective ``ncclCommWindowDeregister`` calls
    stay ordered across ranks even when the subsequent pool drop frees segments
    in allocator-internal order — then drops the ``MemPool`` and reclaims its
    memory (the allocator's free callback runs ``ncclMemFree`` per segment).
    If the CUDA context is poisoned or window cleanup is deferred, the pool is
    retained so a later call can retry without losing ownership.
    """
    idx = _device_index(device)
    with _lock:
        pool = _pools.get(idx)
    allocator = get_backend()
    if pool is None:
        if allocator.has_pending_releases(idx):
            if allocator.should_skip_cleanup(idx):
                logger.warning(
                    "leaving queued symmetric-memory releases on cuda:%d "
                    "after CUDA failure",
                    idx,
                )
                return
            released = _release_pending(allocator, idx)
            logger.info(
                "retried symmetric MemPool cleanup on cuda:%d (released %d segment(s))",
                idx,
                released,
            )
            return
        logger.debug("no symmetric MemPool for cuda:%d; nothing to free", idx)
        return

    if allocator.should_skip_cleanup(idx):
        logger.warning(
            "leaving symmetric MemPool for cuda:%d intact after CUDA failure",
            idx,
        )
        return

    n = _deregister_windows(allocator, idx)
    if n is None:
        return
    if allocator.has_unreleased_resources(idx):
        logger.warning(
            "leaving symmetric MemPool for cuda:%d intact after incomplete "
            "window cleanup",
            idx,
        )
        return

    with _lock:
        if _pools.get(idx) is not pool:
            return
        del _pools[idx]
    del pool
    # empty_cache only releases the current device's segments, so the physical
    # backend free would not run for a non-current device. Dropping the pool
    # runs the allocator's free callback, which only *queues* each segment (the
    # collective deregister is illegal under the caching allocator's lock), so
    # drain the queue here in allocation order.
    _empty_cache(idx)
    released = _release_pending(allocator, idx)
    logger.info(
        "freed symmetric MemPool for cuda:%d (deregistered %d window(s), "
        "released %d segment(s))",
        idx,
        n,
        released,
    )
