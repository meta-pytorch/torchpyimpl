# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from typing import cast

import torch
import torch.distributed as dist
from nccl.bindings import nccl as nccl_bindings
from nccl.core import WindowFlag
from nccl.core.resources import RegisteredWindowHandle

from torchpyimpl.distributed.symmetric_memory._cuda import (
    cu_mem_get_address_range,
    should_skip_cuda_cleanup,
)
from torchpyimpl.distributed.symmetric_memory._symm_mem import (
    _PeerAllocInfo,
    AllocRecord,
    buffer_offset,
    CleanupStatus,
    SymmetricMemoryAllocatorBase,
)


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class _NcclHandle:
    base: int
    device: int
    released: bool = False


@dataclass
class _NcclWindow:
    handle: RegisteredWindowHandle
    group: weakref.ref[dist.ProcessGroup] | None


def _group_alive(group: weakref.ref[dist.ProcessGroup] | None) -> bool:
    if group is None:
        return True
    pg = group()
    return pg is not None and pg in dist.distributed_c10d._world.pg_map


_PeerTables = tuple[list[int], list[int], torch.Tensor, torch.Tensor]


def _peer_tables(
    handle: RegisteredWindowHandle, world_size: int, off: int, device: torch.device
) -> _PeerTables:
    signal_pads: list[int] = []
    for peer_rank in range(world_size):
        p = handle.get_peer_device_pointer(peer_rank)
        if p is None or p == 0:
            raise RuntimeError(
                f"ncclGetPeerDevicePointer returned NULL for peer "
                f"{peer_rank}; peer is not reachable via LSA"
            )
        signal_pads.append(int(p))
    buffers = [s + off for s in signal_pads]
    return (
        signal_pads,
        buffers,
        torch.tensor(signal_pads, dtype=torch.int64, device=device),
        torch.tensor(buffers, dtype=torch.int64, device=device),
    )


class NCCLAllocator(SymmetricMemoryAllocatorBase):
    def alloc_physical(self, block_size: int, device: int) -> tuple[int, object]:
        with torch.cuda.device(device):
            base = int(nccl_bindings.mem_alloc(block_size))
        return base, _NcclHandle(base, device)

    def free_physical(self, handle: object) -> CleanupStatus:
        h = cast(_NcclHandle, handle)
        if h.released:
            return CleanupStatus.CLEANED
        device = h.device
        if should_skip_cuda_cleanup(device):
            return CleanupStatus.DEFERRED
        try:
            with torch.cuda.device(device):
                nccl_bindings.mem_free(h.base)
        except Exception:
            logger.warning("ncclMemFree failed during cleanup", exc_info=True)
            return CleanupStatus.FAILED
        h.released = True
        return CleanupStatus.CLEANED

    def allocation_base(self, ptr: int) -> int:
        return cu_mem_get_address_range(ptr)

    def establish_peers(
        self,
        rec: AllocRecord,
        comm_ptr: int,
        world_size: int,
        device: torch.device,
        group: object | None = None,
    ) -> _PeerAllocInfo:
        dev_idx = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        if dev_idx != rec.device:
            raise ValueError(
                f"rendezvous device cuda:{dev_idx} does not own allocation on "
                f"cuda:{rec.device}"
            )
        pg = cast(dist.ProcessGroup | None, group)
        if world_size > 1 and pg is None:
            raise ValueError(
                "the NCCL backend needs the process group for a rendezvous of "
                "more than one rank"
            )

        handle = RegisteredWindowHandle(
            nccl_bindings.Comm(comm_ptr),
            rec.alloc_base,
            rec.block_size,
            flags=WindowFlag.CollSymmetric,
        )
        off = buffer_offset()
        try:
            mc = handle.get_lsa_multimem_device_pointer(off)
            mc_base = handle.get_lsa_multimem_device_pointer(0)
            multicast_ptr = 0 if mc is None else int(mc)
            multicast_base = 0 if mc_base is None else int(mc_base)
        except Exception:
            logger.warning("no NCCL multimem pointer for this window", exc_info=True)
            multicast_ptr = multicast_base = 0

        local_error: Exception | None = None
        tables: _PeerTables | None = None
        try:
            tables = _peer_tables(handle, world_size, off, device)
        except Exception as e:
            local_error = e
        except BaseException:
            self._close_unpublished(handle)
            raise

        ok, all_multicast = local_error is None, multicast_ptr != 0
        if pg is not None and world_size > 1:
            votes: list[object] = [None] * world_size
            try:
                with torch.cuda.device(rec.device):
                    dist.all_gather_object(votes, (ok, all_multicast), group=pg)
            except BaseException:
                self._close_unpublished(handle)
                raise
            ok = all(cast(tuple[bool, bool], v)[0] for v in votes)
            all_multicast = all(cast(tuple[bool, bool], v)[1] for v in votes)
        if not ok:
            self._close_unpublished(handle)
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                "a peer rank failed to set up the NCCL window during rendezvous; "
                "see that rank's traceback for the cause"
            )
        if not all_multicast:
            multicast_ptr = multicast_base = 0
        signal_pads, buffers, signal_pads_dev, buffers_dev = cast(_PeerTables, tables)
        logger.info(
            "registered symmetric window: base=0x%x total=%d world_size=%d "
            "comm=0x%x multicast=%s",
            rec.alloc_base,
            rec.block_size,
            world_size,
            comm_ptr,
            "yes" if multicast_ptr else "no",
        )
        return _PeerAllocInfo(
            window=_NcclWindow(handle, None if pg is None else weakref.ref(pg)),
            buffers=buffers,
            signal_pads=signal_pads,
            buffers_dev=buffers_dev,
            signal_pads_dev=signal_pads_dev,
            multicast_ptr=multicast_ptr,
            multicast_base=multicast_base,
            device=rec.device,
        )

    def _close_unpublished(self, handle: RegisteredWindowHandle) -> None:
        try:
            handle.close()
        except Exception:
            logger.warning("failed to close unpublished NCCL window", exc_info=True)
            self.retain_unpublished_failure(handle)

    def close_peers(self, pai: _PeerAllocInfo) -> CleanupStatus:
        window = cast(_NcclWindow | None, pai.window)
        if window is None:
            return CleanupStatus.CLEANED
        if pai.cleanup_failed:
            return CleanupStatus.FAILED
        if not _group_alive(window.group):
            pai.window = None
            return CleanupStatus.CLEANED
        if should_skip_cuda_cleanup(pai.device):
            return CleanupStatus.DEFERRED
        try:
            window.handle.close()
        except Exception:
            logger.warning("window.close() failed", exc_info=True)
            return CleanupStatus.FAILED
        pai.window = None
        return CleanupStatus.CLEANED

    def peers_alive(self, pai: _PeerAllocInfo) -> bool:
        window = cast(_NcclWindow | None, pai.window)
        return window is None or _group_alive(window.group)
