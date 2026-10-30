# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA-IPC backend: CUDA VMM allocations shared over fabric handles or POSIX
fds.
"""

from __future__ import annotations

import enum
import functools
import logging
import os
import socket
from dataclasses import dataclass
from typing import cast

import torch
import torch.distributed as dist
from cuda.bindings import driver

from torchpyimpl.distributed.symmetric_memory import _cuda
from torchpyimpl.distributed.symmetric_memory._symm_mem import (
    _device_index,
    _PeerAllocInfo,
    AllocRecord,
    buffer_offset,
    CleanupStatus,
    SymmetricMemoryAllocatorBase,
)
from torchpyimpl.distributed.symmetric_memory.backends import _ipc_fd


logger: logging.Logger = logging.getLogger(__name__)

_TRANSPORT_ENV: str = "TORCHPYIMPL_CUDA_IPC_TRANSPORT"

# Resolved once per process: the handle type is baked into each block.
_transport: IpcHandleType | None = None


def get_transport(device: int) -> IpcHandleType:
    """The IPC handle type this process exports, resolved once.

    ``TORCHPYIMPL_CUDA_IPC_TRANSPORT`` forces ``fabric`` or ``fd``; otherwise prefer
    fabric (it is multi-node capable and needs no fd passing) and fall back to
    POSIX fds."""
    global _transport
    if _transport is not None:
        return _transport
    transport = _resolve_transport(device)
    _transport = transport
    logger.info(
        "CUDA-IPC transport: %s (fabric=%s, posix_fd=%s)",
        transport.value,
        cu_device_supports_fabric(device),
        cu_device_supports_posix_fd(device),
    )
    return transport


def _resolve_transport(device: int) -> IpcHandleType:
    requested = os.environ.get(_TRANSPORT_ENV, "auto").lower()
    if requested not in ("auto", "fabric", "fd"):
        raise ValueError(
            f"{_TRANSPORT_ENV} must be one of 'auto', 'fabric', 'fd'; got {requested!r}"
        )
    if requested == "fabric":
        _require(cu_device_supports_fabric(device), device, "fabric")
        return IpcHandleType.FABRIC
    if requested == "fd":
        _require(cu_device_supports_posix_fd(device), device, "POSIX-fd")
        return IpcHandleType.POSIX_FD
    if cu_device_supports_fabric(device):
        return IpcHandleType.FABRIC
    _require(cu_device_supports_posix_fd(device), device, "POSIX-fd")
    return IpcHandleType.POSIX_FD


def _require(supported: bool, device: int, what: str) -> None:
    if not supported:
        raise NotImplementedError(
            f"the CUDA-IPC backend needs {what} handle support, which device "
            f"{device} does not report."
        )


def reset_transport() -> None:
    """Forget the resolved transport (tests that flip the env var)."""
    global _transport
    _transport = None


# === CUDA VMM / IPC driver wrappers ==========================================
#
# Handle types (fixed per process, baked into each block at cuMemCreate):
# FABRIC is 64 bytes exchanged with all_gather_object, multi-node; POSIX_FD is an
# fd passed with SCM_RIGHTS (see _ipc_fd), single node.

_FABRIC: object = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
_POSIX_FD: object = (
    driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
)


class IpcHandleType(enum.Enum):
    """Which CUDA shareable-handle kind the CUDA-IPC backend exports."""

    FABRIC = "fabric"
    POSIX_FD = "fd"


def _driver_handle_type(kind: IpcHandleType) -> object:
    return _FABRIC if kind is IpcHandleType.FABRIC else _POSIX_FD


# Device count used only to query the multicast granularity at allocation time.
_MC_GRANULARITY_DEVICES: int = 2


def _check(err: object, what: str) -> None:
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {err}")


def _round_up(n: int, alignment: int) -> int:
    # Arithmetic (not bit-mask) rounding: the VMM granularity need not be a power
    # of two, though in practice it is (2 MiB on GB200).
    return (n + alignment - 1) // alignment * alignment


def _alloc_prop(device: int, kind: IpcHandleType) -> object:
    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.requestedHandleTypes = _driver_handle_type(kind)
    return prop


def _rw_access_desc(device: int) -> object:
    desc = driver.CUmemAccessDesc()
    desc.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device
    desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return desc


def _map_rw(base: int, size: int, handle: object, device: int) -> None:
    """Map ``handle`` at reserved VA ``base`` and grant this device RW access."""
    (err,) = driver.cuMemMap(base, size, 0, handle, 0)
    _check(err, "cuMemMap")
    (err,) = driver.cuMemSetAccess(base, size, [_rw_access_desc(device)], 1)
    _check(err, "cuMemSetAccess")


@functools.cache
def cu_device_supports_fabric(device: int) -> bool:
    """Whether ``device`` can export/import ``CU_MEM_HANDLE_TYPE_FABRIC`` handles
    (true on GB200/GB300-class NVLink fabric, and on IMEX-enabled H100 domains).

    Cached: the attribute is static per device, and ``alloc_physical`` asks on
    every block from inside the caching allocator's alloc callback."""
    _cuda._init()
    attr = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
    err, val = driver.cuDeviceGetAttribute(attr, device)
    return err == driver.CUresult.CUDA_SUCCESS and int(val) == 1


@functools.cache
def cu_device_supports_posix_fd(device: int) -> bool:
    """Whether ``device`` can export/import POSIX file-descriptor handles — the
    single-node fallback for hardware without fabric support. True on essentially
    every Linux CUDA device."""
    _cuda._init()
    attr = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED
    err, val = driver.cuDeviceGetAttribute(attr, device)
    return err == driver.CUresult.CUDA_SUCCESS and int(val) == 1


def cu_map_handle(handle: object, size: int, device: int) -> int:
    """Reserve a VA range, map ``handle`` into it, grant this device RW access,
    and return the mapped base pointer. Works for both a memory handle and a
    multicast object handle.

    Owns exactly the VA reservation it makes: if the map fails, that reservation
    is released before re-raising, so a failed map cannot strand address space.
    ``handle`` stays the caller's to release — this is also the multicast path,
    where the object outlives any one mapping."""
    err, reserved = driver.cuMemAddressReserve(size, 0, 0, 0)
    _check(err, "cuMemAddressReserve")
    base = int(reserved)
    try:
        _map_rw(base, size, handle, device)
    except Exception as e:
        driver.cuMemUnmap(base, size)
        driver.cuMemAddressFree(base, size)
        raise e
    return base


def cu_vmm_alloc(
    min_size: int, device: int, kind: IpcHandleType = IpcHandleType.FABRIC
) -> tuple[int, int, object]:
    """``cuMemCreate`` a VMM block of at least ``min_size`` bytes exportable as
    ``kind``, map it RW on ``device``, and return
    ``(base_ptr, mapped_size, handle)``.

    ``kind`` is baked into the block here (``requestedHandleTypes``): a block
    created for FABRIC cannot later be exported as a POSIX fd, or vice versa.

    The size is rounded up to the driver's recommended allocation granularity —
    and, where multicast is supported, to the multicast granularity too, so the
    block can later be bound to a multicast object (``cuMulticastBindMem`` needs
    a size that is a multiple of the multicast granularity). ``mapped_size`` is
    what every rank must map, so it — not the caller's request — is what peers
    reserve/map on import."""
    _cuda._init()
    prop = _alloc_prop(device, kind)
    err, gran = driver.cuMemGetAllocationGranularity(
        prop,
        driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
    )
    _check(err, "cuMemGetAllocationGranularity")
    granularity = int(gran)
    if cu_device_supports_multicast(device):
        # The world size is not known until rendezvous, so the granularity is
        # queried for _MC_GRANULARITY_DEVICES. It is device-count independent on
        # current hardware, and if that ever stops holding the size is simply
        # rejected at cuMulticastBindMem, which _init_multicast degrades to the
        # peer-to-peer path rather than failing the rendezvous.
        granularity = max(
            granularity, cu_multicast_granularity(_MC_GRANULARITY_DEVICES, kind)
        )
    size = _round_up(min_size, granularity)
    err, handle = driver.cuMemCreate(size, prop, 0)
    _check(err, "cuMemCreate")
    try:
        base = cu_map_handle(handle, size, device)
    except Exception as e:
        driver.cuMemRelease(handle)  # we created it; no one else can free it
        raise e
    return base, size, handle


def cu_mem_export(
    handle: object, kind: IpcHandleType = IpcHandleType.FABRIC
) -> bytes | int:
    """Export ``handle`` (a memory or multicast object handle) as a shareable
    handle of ``kind``:

    * FABRIC → 64 ``bytes``, picklable, ships over ``dist.all_gather_object``.
    * POSIX_FD → an ``int`` file descriptor, **owned by the caller**: it must be
      passed to peers with ``SCM_RIGHTS`` and then closed. Each call returns a
      fresh fd.
    """
    err, sh = driver.cuMemExportToShareableHandle(handle, _driver_handle_type(kind), 0)
    _check(err, "cuMemExportToShareableHandle")
    if kind is IpcHandleType.FABRIC:
        return bytes(sh.data)
    return int(sh)


def cu_mem_import(
    raw: bytes | int, kind: IpcHandleType = IpcHandleType.FABRIC
) -> object:
    """Import a peer's shareable handle (memory or multicast). ``raw`` goes
    straight to ``cuMemImportFromShareableHandle`` as the ``void*`` ``osHandle``
    — the raw 64 bytes for FABRIC, the fd (as an int) for POSIX_FD. The driver
    dups a POSIX fd internally, so the caller still owns and must close it."""
    _cuda._init()
    err, handle = driver.cuMemImportFromShareableHandle(raw, _driver_handle_type(kind))
    _check(err, "cuMemImportFromShareableHandle")
    return handle


def cu_vmm_import_map(
    raw: bytes | int,
    size: int,
    device: int,
    kind: IpcHandleType = IpcHandleType.FABRIC,
) -> tuple[int, object]:
    """Import a peer's memory handle and map it RW on ``device``; return
    ``(base_ptr, handle)``."""
    handle = cu_mem_import(raw, kind)
    try:
        return cu_map_handle(handle, size, device), handle
    except Exception as e:
        cu_mem_release(handle)  # we imported it; the caller never saw it
        raise e


def cu_mem_unmap(base: int, size: int) -> bool:
    """``cuMemUnmap`` the VA range and free the reservation. Kept separate from
    the physical release so teardown can interleave a multicast unbind in
    between (unbind must precede the physical release).

    Both steps run even if the first failed — teardown must not raise part-way
    and leak the rest — but each failure is logged rather than dropped, so a
    leaked mapping leaves a trace instead of vanishing."""
    ok = True
    for what, step in (
        ("cuMemUnmap", lambda: driver.cuMemUnmap(base, size)),
        ("cuMemAddressFree", lambda: driver.cuMemAddressFree(base, size)),
    ):
        try:
            (err,) = step()
        except Exception:
            logger.warning(
                "%s raised while unmapping 0x%x (%d bytes)",
                what,
                base,
                size,
                exc_info=True,
            )
            ok = False
            continue
        if err != driver.CUresult.CUDA_SUCCESS:
            logger.warning(
                "%s failed unmapping 0x%x (%d bytes): %s", what, base, size, err
            )
            ok = False
    return ok


def cu_mem_release(handle: object) -> bool:
    """Release a physical memory / multicast object handle. Logs rather than
    raises: callers are teardown paths that must keep going."""
    try:
        (err,) = driver.cuMemRelease(handle)
    except Exception:
        logger.warning("cuMemRelease raised during cleanup", exc_info=True)
        return False
    if err != driver.CUresult.CUDA_SUCCESS:
        logger.warning("cuMemRelease failed: %s", err)
        return False
    return True


def cu_vmm_free(base: int, size: int, handle: object) -> bool:
    """Unmap the VA (+ free the reservation) and release the physical handle."""
    unmapped = cu_mem_unmap(base, size)
    released = cu_mem_release(handle)
    return unmapped and released


# === CUDA multicast (multimem) ================================================
#
# Rank 0 creates and exports the object; every rank imports it, adds its device,
# binds its block, and maps it.

_MULTICAST_ATTR: object = (
    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
)


@functools.cache
def cu_device_supports_multicast(device: int) -> bool:
    """Whether ``device`` supports multicast objects (needed for the multimem /
    LSA pointer)."""
    _cuda._init()
    err, val = driver.cuDeviceGetAttribute(_MULTICAST_ATTR, device)
    return err == driver.CUresult.CUDA_SUCCESS and int(val) == 1


def _multicast_prop(num_devices: int, size: int, kind: IpcHandleType) -> object:
    prop = driver.CUmulticastObjectProp()
    prop.numDevices = num_devices
    prop.size = size
    prop.handleTypes = _driver_handle_type(kind)
    prop.flags = 0
    return prop


@functools.cache
def cu_multicast_granularity(
    num_devices: int, kind: IpcHandleType = IpcHandleType.FABRIC
) -> int:
    """Minimum granularity a multicast object — and the memory bound to it — must
    be a multiple of."""
    _cuda._init()
    prop = _multicast_prop(num_devices, 0, kind)
    err, gran = driver.cuMulticastGetGranularity(
        prop, driver.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_MINIMUM
    )
    _check(err, "cuMulticastGetGranularity")
    return int(gran)


def cu_multicast_create(
    size: int, num_devices: int, kind: IpcHandleType = IpcHandleType.FABRIC
) -> object:
    """Create a multicast object spanning ``num_devices`` (>= 2) of ``size``
    bytes, exportable as ``kind``. Only one rank creates it; the others import
    the handle it shares."""
    _cuda._init()
    prop = _multicast_prop(num_devices, size, kind)
    err, mc = driver.cuMulticastCreate(prop)
    _check(err, "cuMulticastCreate")
    return mc


def cu_multicast_add_device(mc: object, device: int) -> None:
    (err,) = driver.cuMulticastAddDevice(mc, device)
    _check(err, "cuMulticastAddDevice")


def cu_multicast_bind_mem(mc: object, mem_handle: object, size: int) -> None:
    """Bind ``mem_handle`` (this rank's physical block) into ``mc`` at offset 0."""
    (err,) = driver.cuMulticastBindMem(mc, 0, mem_handle, 0, size, 0)
    _check(err, "cuMulticastBindMem")


def cu_multicast_unbind(mc: object, device: int, size: int) -> None:
    """Unbind ``device``'s memory from ``mc`` (before the block is released)."""
    (err,) = driver.cuMulticastUnbind(mc, device, 0, size)
    _check(err, "cuMulticastUnbind")


@dataclass
class _CudaMapping:
    base: int  # mapped VA base (== signal-pad base, signal-pad-first)
    size: int  # granularity-rounded mapped size (all ranks agree on this)
    handle: object  # CUmemGenericAllocationHandle owning / referencing the memory
    device: int


@dataclass
class _CudaMulticast:
    """The multicast object for one (allocation, group): the multimem mapping +
    the handle to unbind/release at teardown."""

    handle: object  # the multicast object handle (owns the multimem)
    addr: int  # mapped multicast (multimem) VA base
    size: int
    device: int


@dataclass
class _CudaPeerState:
    """Backend state stored in ``_PeerAllocInfo.window`` for one (allocation,
    group): the imported peer mappings and the multicast object, if any. Torn
    down together in :meth:`CUDAIPCAllocator.close_peers`."""

    imported: list[_CudaMapping]
    multicast: _CudaMulticast | None


class CUDAIPCAllocator(SymmetricMemoryAllocatorBase):
    requires_nccl_comm: bool = False

    def alloc_physical(self, block_size: int, device: int) -> tuple[int, object]:
        transport = get_transport(device)
        base, size, handle = cu_vmm_alloc(block_size, device, transport)
        return base, _CudaMapping(base=base, size=size, handle=handle, device=device)

    def free_physical(self, handle: object) -> CleanupStatus:
        m = cast(_CudaMapping, handle)
        # Synchronize the device that *owns* the mapping, not whichever happens
        # to be current: free_pool calls release_pending outside its
        # torch.cuda.device block, so unmapping under the wrong context would
        # pull memory out from under kernels still running on m.device.
        if _cuda.should_skip_cuda_cleanup(m.device):
            return CleanupStatus.DEFERRED
        with torch.cuda.device(m.device):
            if not cu_vmm_free(m.base, m.size, m.handle):
                return CleanupStatus.FAILED
        return CleanupStatus.CLEANED

    def allocation_base(self, ptr: int) -> int:
        # cuMemCreate is VMM-backed, so the driver knows the allocation range —
        # same O(1) interior-pointer recovery as the NCCL backend.
        return _cuda.cu_mem_get_address_range(ptr)

    def establish_peers(
        self,
        rec: AllocRecord,
        comm_ptr: int,
        world_size: int,
        device: torch.device,
        group: object | None = None,
    ) -> _PeerAllocInfo:
        if group is None:
            raise RuntimeError(
                "CUDA-IPC rendezvous needs the process group to exchange IPC "
                "handles; call rendezvous(tensor, group) (the group is threaded "
                "through to the allocator)."
            )
        pg = cast(dist.ProcessGroup, group)
        rank = dist.get_rank(pg)
        mine = cast(_CudaMapping, rec.handle)
        transport = get_transport(mine.device)

        # All-gather every rank's (size, device, host, multicast-support), plus
        # the shareable handle itself when it is picklable (fabric). The size
        # travels so importers map exactly what the exporter created; host+device
        # catch two ranks colliding on one GPU (and, for fd, a multi-host group);
        # multicast gates the multimem path on unanimity.
        payload = {
            "handle": (
                cu_mem_export(mine.handle, transport)
                if transport is IpcHandleType.FABRIC
                else None
            ),
            "size": mine.size,
            "device": mine.device,
            "host": socket.gethostname(),
            "multicast": cu_device_supports_multicast(mine.device),
            "transport": transport.value,
        }
        gathered_objects: list[object] = [None] * world_size
        dist.all_gather_object(gathered_objects, payload, group=pg)
        gathered = [cast(dict[str, object], info) for info in gathered_objects]
        self._validate(gathered, rank, mine.size, transport)

        off = buffer_offset()
        dev_idx = _device_index(device)

        exported = self._agree_ready_to_exchange(
            mine, dev_idx, transport, pg, world_size
        )

        signal_pads: list[int] = []
        imported: list[_CudaMapping] = []
        buffers: list[int] = []
        buffers_dev: torch.Tensor | None = None
        signal_pads_dev: torch.Tensor | None = None
        setup_error: Exception | None = None
        try:
            signal_pads, imported = self._import_peers(
                mine, gathered, pg, rank, world_size, dev_idx, transport, exported
            )
            buffers = [s + off for s in signal_pads]
            buffers_dev = torch.tensor(buffers, dtype=torch.int64, device=device)
            signal_pads_dev = torch.tensor(
                signal_pads, dtype=torch.int64, device=device
            )
        except Exception as e:
            # Do not leave yet. Every rank is about to enter _init_multicast's
            # collectives, and a rank that walked out here would leave the rest
            # blocked in the first one until the group times out. Record the
            # failure, vote it below, then raise. _import_peers rolls back its
            # own partial failure; the vote-failure path below releases mappings
            # from ranks that completed the import.
            setup_error = e

        if not _all_agree(pg, world_size, setup_error is None):
            for m in imported:
                cu_vmm_free(m.base, m.size, m.handle)
            if setup_error is not None:
                raise setup_error
            raise RuntimeError(
                "a peer rank failed to prepare symmetric memory during "
                "rendezvous; see that rank's traceback for the cause"
            )

        if buffers_dev is None or signal_pads_dev is None:
            raise RuntimeError("peer pointer tensors were not initialized")

        try:
            multicast, multicast_ptr = self._init_multicast(
                mine, gathered, pg, rank, world_size, dev_idx, off, transport
            )
        except Exception as e:
            # _init_multicast tears down its own multicast state, but a raise
            # here would still strand every peer mapping imported above.
            for m in imported:
                cu_vmm_free(m.base, m.size, m.handle)
            raise e
        logger.info(
            "established CUDA-IPC peers: base=0x%x size=%d world_size=%d "
            "(imported %d peer mapping(s), multicast=%s)",
            mine.base,
            mine.size,
            world_size,
            len(imported),
            "yes" if multicast is not None else "no",
        )
        return _PeerAllocInfo(
            window=_CudaPeerState(imported=imported, multicast=multicast),
            buffers=buffers,
            signal_pads=signal_pads,
            buffers_dev=buffers_dev,
            signal_pads_dev=signal_pads_dev,
            multicast_ptr=multicast_ptr,
            multicast_base=multicast.addr if multicast is not None else 0,
            device=dev_idx,
        )

    @staticmethod
    def _agree_ready_to_exchange(
        mine: _CudaMapping,
        dev_idx: int,
        transport: IpcHandleType,
        pg: dist.ProcessGroup,
        world_size: int,
    ) -> int | None:
        """Do the per-rank work that has to precede the handle exchange, and
        agree with the peers that every rank managed it. Returns this rank's
        exported handle — ``None`` on fabric, where it already travelled in the
        rendezvous payload.

        Everything past the payload gather that can fail on one rank alone
        belongs here. The fd export especially: it sits in front of the
        SCM_RIGHTS ring, which is itself collective, so a rank that skipped it
        would meet its peers' ring gather with this vote instead, and two
        mismatched object collectives pair up rather than erroring. Raising on
        every rank keeps nobody alone in the exchange that follows."""
        local_error: Exception | None = None
        exported: int | None = None
        try:
            if dev_idx != mine.device:
                # Peers get RW access granted to dev_idx, so a mismatch would map
                # this allocation for a GPU that does not own it. _validate cannot
                # see this: it compares the *gathered* devices, which all agree.
                raise RuntimeError(
                    f"rendezvous device cuda:{dev_idx} does not own this "
                    f"allocation (allocated on cuda:{mine.device}); rendezvous "
                    "the tensor on the device it was allocated on."
                )
            if transport is IpcHandleType.POSIX_FD:
                exported = cast(int, cu_mem_export(mine.handle, transport))
        except Exception as e:
            local_error = e

        try:
            agreed = _all_agree(pg, world_size, local_error is None)
        except BaseException as e:
            # The vote can raise outright (group timeout or abort) rather than
            # report, and the export would escape with it.
            if exported is not None:
                os.close(exported)
            raise e
        if agreed:
            return exported
        if exported is not None:
            os.close(exported)
        if local_error is not None:
            raise local_error
        raise RuntimeError(
            "a peer rank failed before the handle exchange during rendezvous; "
            "see that rank's traceback for the cause"
        )

    @staticmethod
    def _import_peers(
        mine: _CudaMapping,
        gathered: list[dict[str, object]],
        pg: dist.ProcessGroup,
        rank: int,
        world_size: int,
        dev_idx: int,
        transport: IpcHandleType,
        exported: int | None,
    ) -> tuple[list[int], list[_CudaMapping]]:
        """Import + map every peer's block; return ``(signal_pad_bases,
        imported_mappings)``. ``signal_pad_bases`` is indexed by rank and holds
        our own mapping at ``rank`` (signal-pad-first, so a block's base *is* its
        signal-pad base)."""
        handles, fds = _share_handles(
            exported, gathered, pg, rank, world_size, transport
        )
        signal_pads: list[int] = []
        imported: list[_CudaMapping] = []
        try:
            for peer_rank in range(world_size):
                if peer_rank == rank:
                    signal_pads.append(mine.base)  # peer==self: our own mapping
                    continue
                size = cast(int, gathered[peer_rank]["size"])
                raw = handles[peer_rank]
                if raw is None:
                    raise RuntimeError(f"missing IPC handle for rank {peer_rank}")
                base, handle = cu_vmm_import_map(
                    raw,
                    size,
                    dev_idx,
                    transport,
                )
                signal_pads.append(base)
                imported.append(
                    _CudaMapping(base=base, size=size, handle=handle, device=dev_idx)
                )
            return signal_pads, imported
        except Exception as e:
            # Nothing owns these yet — no _PeerAllocInfo has been handed back —
            # so a failure partway through would leak every peer imported so far.
            for m in imported:
                cu_vmm_free(m.base, m.size, m.handle)
            raise e
        finally:
            _ipc_fd.close_fds(fds)

    @staticmethod
    def _teardown_multicast(
        mc_handle: object, bound: bool, mc_addr: int, size: int, device: int
    ) -> bool:
        """Undo a partially built multicast object, in reverse order of
        construction. Best-effort throughout: this only runs on paths that are
        already failing, and bailing out early would leak the remainder."""
        ok = True
        if mc_addr:
            ok = cu_mem_unmap(mc_addr, size)
        if bound:
            try:
                cu_multicast_unbind(mc_handle, device, size)
            except Exception:
                logger.warning("cuMulticastUnbind failed during cleanup", exc_info=True)
                ok = False
        if mc_handle is not None:
            ok = cu_mem_release(mc_handle) and ok
        return ok

    def _init_multicast(
        self,
        mine: _CudaMapping,
        gathered: list[dict[str, object]],
        pg: dist.ProcessGroup,
        rank: int,
        world_size: int,
        dev_idx: int,
        off: int,
        transport: IpcHandleType,
    ) -> tuple[_CudaMulticast | None, int]:
        """Establish the multicast (multimem) object for this allocation, or
        ``(None, 0)`` if it is unavailable or any rank failed to set it up.

        Every step here can fail locally while its peers succeed, so each rank
        agrees with its peers before taking a step that blocks on them, and success is
        published only once the object is *mapped*::

            rank 0: create ───────────> agree the object exists
            exchange the handle
            import + add_device ──────> agree before anyone binds
            bind + map ───────────────> agree before anyone commits

        The add_device agreement is what stops a failure becoming a hang:
        ``cuMulticastBindMem`` blocks until every device has been added, so one
        rank dying in ``add_device`` would wedge the rest in bind. The last one
        stops a failure becoming silent divergence: a bound-but-unmapped object
        is an intermediate state peers cannot use, so committing at bind would
        let a rank whose map failed disagree with peers that think multicast is
        live.

        Multicast is an optimization over the peer-to-peer path, so an abandoned
        setup is torn down and reported as unavailable rather than raised."""
        if world_size < 2 or not all(g["multicast"] for g in gathered):
            return None, 0  # need >= 2 devices and unanimous multicast support

        size = mine.size
        mc_handle: object = None
        bound = False
        mc_addr = 0

        def all_agree(ok: bool) -> bool:
            return _all_agree(pg, world_size, ok)

        try:
            # Rank 0 creates the object and everyone agrees it exists before the
            # handle exchange. Torch folds this into the broadcast by sending an
            # invalidator handle; we cannot, because on the POSIX-fd transport
            # the exchange is an SCM_RIGHTS round trip with no way to say "no
            # handle" — hence one collective more than upstream.
            mc_exported: bytes | int | None = None
            if rank == 0:
                try:
                    mc_handle = cu_multicast_create(size, world_size, transport)
                    mc_exported = cu_mem_export(mc_handle, transport)
                except Exception:
                    logger.warning("multicast create/export failed", exc_info=True)
            # The export is inside the guard and inside this vote: the exchange
            # below is collective, so rank 0 failing to export has to degrade
            # multicast for everyone rather than leave peers waiting in it.
            if not all_agree(rank != 0 or mc_exported is not None):
                self._teardown_multicast(mc_handle, bound, mc_addr, size, dev_idx)
                return None, 0

            # Only rank 0 has anything to share; everyone reads index 0.
            shared, fds = _broadcast_handle(
                mc_exported, pg, rank, world_size, transport
            )
            try:
                if shared is None:
                    raise RuntimeError("multicast handle exchange returned no handle")
                if rank != 0:
                    mc_handle = cu_mem_import(shared, transport)
                cu_multicast_add_device(mc_handle, dev_idx)
                local_ok = True
            except Exception:
                logger.warning("multicast import/add_device failed", exc_info=True)
                local_ok = False
            finally:
                _ipc_fd.close_fds(fds)
            if not all_agree(local_ok):
                self._teardown_multicast(mc_handle, bound, mc_addr, size, dev_idx)
                return None, 0

            try:
                cu_multicast_bind_mem(mc_handle, mine.handle, size)
                bound = True
                mc_addr = cu_map_handle(mc_handle, size, dev_idx)
                local_ok = True
            except Exception:
                logger.warning("multicast bind/map failed", exc_info=True)
                local_ok = False
            if not all_agree(local_ok):
                self._teardown_multicast(mc_handle, bound, mc_addr, size, dev_idx)
                return None, 0

            return (
                _CudaMulticast(
                    handle=mc_handle, addr=mc_addr, size=size, device=dev_idx
                ),
                mc_addr + off,
            )
        except Exception as e:
            # The staged returns above only cover failure a collective
            # *reports*. A collective can also raise outright — a group timeout
            # or abort — and whatever was built so far would leak.
            self._teardown_multicast(mc_handle, bound, mc_addr, size, dev_idx)
            raise e

    def close_peers(self, pai: _PeerAllocInfo) -> CleanupStatus:
        state = cast(_CudaPeerState | None, pai.window)
        if state is None:
            return CleanupStatus.CLEANED
        mc = state.multicast
        if mc is None and not state.imported:
            pai.window = None
            return CleanupStatus.CLEANED
        # The guard synchronizes the device that *owns* these mappings, not
        # whichever happens to be current: free_pool drains outside its
        # torch.cuda.device block, and unmapping under the wrong context leaves
        # kernels on the owning device reading memory that is already gone.
        owner = mc.device if mc is not None else state.imported[0].device
        if _cuda.should_skip_cuda_cleanup(owner):
            return CleanupStatus.DEFERRED
        ok = True
        with torch.cuda.device(owner):
            # Multicast first: unmap + unbind + release BEFORE the bound block is
            # released (in free_physical) — cuMulticastUnbind must see it alive.
            if mc is not None:
                ok = self._teardown_multicast(
                    mc.handle, True, mc.addr, mc.size, mc.device
                )
            # cu_vmm_free logs and continues, so one bad mapping cannot abandon
            # the rest of the batch.
            for m in state.imported:
                ok = cu_vmm_free(m.base, m.size, m.handle) and ok
        if not ok:
            return CleanupStatus.FAILED
        pai.window = None
        return CleanupStatus.CLEANED

    @staticmethod
    def _validate(
        gathered: list[dict[str, object]],
        rank: int,
        my_size: int,
        transport: IpcHandleType,
    ) -> None:
        """Fail loudly on the cross-rank mismatches that would otherwise corrupt
        symmetric memory silently: unequal block sizes (ranks would map different
        extents), ranks that resolved different transports (their exchanges do
        not interoperate), two ranks sharing one physical GPU, and — for the fd
        transport — a group that spans hosts."""
        kinds = {str(info["transport"]) for info in gathered}
        if len(kinds) > 1:
            raise RuntimeError(
                f"ranks disagree on the CUDA-IPC transport ({sorted(kinds)}). "
                "It is resolved per process from local device attributes and "
                "TORCHPYIMPL_CUDA_IPC_TRANSPORT, and the two exchanges are not "
                "interchangeable — set the env var identically on every rank, "
                "or use a group whose devices all support the same handle type."
            )

        seen: set[tuple[object, object]] = set()
        for peer_rank, info in enumerate(gathered):
            if info["size"] != my_size:
                raise RuntimeError(
                    f"symmetric block-size mismatch: rank {rank} has {my_size} "
                    f"bytes, peer {peer_rank} has {info['size']}. Every rank must "
                    "rendezvous the matching allocation."
                )
            key = (info["host"], info["device"])
            if key in seen:
                raise RuntimeError(
                    f"two ranks report the same GPU {key}; symmetric memory needs "
                    "exactly one rank per device (check LOCAL_RANK / set_device)."
                )
            seen.add(key)

        if transport is not IpcHandleType.POSIX_FD:
            return
        hosts = {str(info["host"]) for info in gathered}
        if len(hosts) > 1:
            raise RuntimeError(
                f"the POSIX-fd IPC transport is single-node only, but this group "
                f"spans {len(hosts)} hosts ({sorted(hosts)}). File descriptors "
                "cannot cross hosts. Use fabric-capable hardware, or restrict the "
                "symmetric-memory group to one node."
            )


def _all_agree(pg: dist.ProcessGroup, world_size: int, ok: bool) -> bool:
    """Collective AND over ``ok``.

    Every rank must reach this even after a local failure — that is the whole
    point. A rank that returns early instead leaves its peers inside the next
    collective, where a later mismatched one pairs up with it and the ranks
    silently diverge."""
    flags: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(flags, ok, group=pg)
    return all(flags)


def _share_handles(
    exported: int | None,
    gathered: list[dict[str, object]],
    pg: dist.ProcessGroup,
    rank: int,
    world_size: int,
    transport: IpcHandleType,
) -> tuple[list[bytes | int | None], list[int | None]]:
    """Every rank's shareable handle for its own block, indexed by rank, plus the
    fds the caller must close once they are imported (empty for fabric).

    Fabric handles already rode ``all_gather_object`` inside ``gathered``; fds
    cannot be pickled, so they go around the ``SCM_RIGHTS`` ring instead.

    ``exported`` is already exported by the caller, which votes on that
    succeeding first: the ring below is collective, so a rank that failed to
    export must not be the one rank missing from it."""
    if transport is IpcHandleType.FABRIC:
        handles = [cast(bytes | int | None, info["handle"]) for info in gathered]
        return handles, []
    try:
        fds = _ipc_fd.all_gather_fds(exported, pg)
    except BaseException as e:
        if exported is not None:
            os.close(exported)
        raise e
    if any(fd is None for fd in fds):
        # Every rank contributes a real fd here, so a gap is a transport bug.
        # Catching it now beats handing None to cuMemImportFromShareableHandle.
        missing = [i for i, fd in enumerate(fds) if fd is None]
        _ipc_fd.close_fds(fds)
        raise RuntimeError(f"fd exchange lost the handle from rank(s) {missing}")
    handles: list[bytes | int | None] = list(fds)
    return handles, fds


def _broadcast_handle(
    exported: bytes | int | None,
    pg: dist.ProcessGroup,
    rank: int,
    world_size: int,
    transport: IpcHandleType,
) -> tuple[bytes | int | None, list[int | None]]:
    """Share rank 0's already-exported multicast handle with every rank. Returns
    it plus the fds to close afterwards.

    The caller exports and votes on that succeeding, because this function is
    collective — rank 0 failing here would strand every peer inside it."""
    if transport is IpcHandleType.FABRIC:
        shared: list[object] = [None for _ in range(world_size)]
        dist.all_gather_object(shared, exported, group=pg)
        return cast(bytes | int | None, shared[0]), []
    fd = cast(int | None, exported)
    try:
        fds = _ipc_fd.all_gather_fds(fd, pg)
    except BaseException as e:
        if fd is not None:
            os.close(fd)
        raise e
    if fds[0] is None:
        # Only the root contributes here, so an empty root slot means the ring
        # lost it — a transport bug, not the caller declining to export.
        _ipc_fd.close_fds(fds)
        raise RuntimeError("fd exchange lost the multicast handle from rank 0")
    return fds[0], fds
