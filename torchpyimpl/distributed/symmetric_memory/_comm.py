# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
import threading
import weakref
from dataclasses import dataclass

import torch
import torch.distributed as dist

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupInfo:
    comm_ptr: int
    rank: int
    world_size: int
    timeout_ms: int


@dataclass(frozen=True)
class _CacheEntry:
    info: GroupInfo
    group_ref: "weakref.ref[dist.ProcessGroup]"


_group_cache: dict[tuple[int, int, bool], _CacheEntry] = {}


def clear_comm_ptr_cache() -> None:
    _group_cache.clear()


def _timeout_ms(pg: dist.ProcessGroup, device: torch.device) -> int:
    try:
        timeout = pg._get_backend(device).options._timeout
    except AttributeError:
        logger.debug(
            "%s exposes no options._timeout; signal spins stay unbounded",
            type(pg).__name__,
        )
        return 0
    return int(timeout.total_seconds() * 1000)


def get_nccl_comm_ptr(pg: dist.ProcessGroup, device: torch.device) -> int:
    return get_group_info(pg, device).comm_ptr


_synthetic_ids: "weakref.WeakKeyDictionary[dist.ProcessGroup, dict[int, int]]" = (
    weakref.WeakKeyDictionary()
)
_next_synthetic_id: int = 0
_synthetic_lock: threading.Lock = threading.Lock()


def _synthetic_group_id(pg: dist.ProcessGroup, dev_index: int) -> int:
    global _next_synthetic_id
    with _synthetic_lock:
        per_device = _synthetic_ids.setdefault(pg, {})
        existing = per_device.get(dev_index)
        if existing is not None:
            return existing
        _next_synthetic_id -= 1
        per_device[dev_index] = _next_synthetic_id
        return _next_synthetic_id


def _resolve_comm_ptr(pg: dist.ProcessGroup, device: torch.device) -> int:
    backend = dist.get_backend_impl(pg, device)
    ptr = int(backend.comm_ptr)  # pyre-ignore[16]
    if ptr == 0:
        raise RuntimeError(
            f"{type(backend).__name__} returned a null comm pointer for device "
            f"{device}."
        )
    logger.debug("resolved ncclComm_t for device %s: 0x%x", device, ptr)
    return ptr


def get_group_info(
    pg: dist.ProcessGroup, device: torch.device, *, require_comm: bool = True
) -> GroupInfo:
    dev_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    key = (id(pg), dev_index, require_comm)
    cached = _group_cache.get(key)
    if cached is not None and cached.group_ref() is pg:
        return cached.info

    torch.cuda.set_device(device)

    ptr = (
        _resolve_comm_ptr(pg, device)
        if require_comm
        else _synthetic_group_id(pg, dev_index)
    )
    info = GroupInfo(
        comm_ptr=ptr,
        rank=dist.get_rank(pg),
        world_size=dist.get_world_size(pg),
        timeout_ms=_timeout_ms(pg, device),
    )
    _group_cache[key] = _CacheEntry(info, weakref.ref(pg))
    return info
