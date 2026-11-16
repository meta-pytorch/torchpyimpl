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

_transport: IpcHandleType | None = None


def get_transport(device: int) -> IpcHandleType:
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
    global _transport
    _transport = None


_FABRIC: object = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
_POSIX_FD: object = (
    driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
)


class IpcHandleType(enum.Enum):
    FABRIC = "fabric"
    POSIX_FD = "fd"


def _driver_handle_type(kind: IpcHandleType) -> object:
    return _FABRIC if kind is IpcHandleType.FABRIC else _POSIX_FD


_MC_GRANULARITY_DEVICES: int = 2


def _check(err: object, what: str) -> None:
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {err}")


def _round_up(n: int, alignment: int) -> int:
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
    (err,) = driver.cuMemMap(base, size, 0, handle, 0)
    _check(err, "cuMemMap")
    (err,) = driver.cuMemSetAccess(base, size, [_rw_access_desc(device)], 1)
    _check(err, "cuMemSetAccess")


@functools.cache
def cu_device_supports_fabric(device: int) -> bool:
    _cuda._init()
    attr = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
    err, val = driver.cuDeviceGetAttribute(attr, device)
    return err == driver.CUresult.CUDA_SUCCESS and int(val) == 1


@functools.cache
def cu_device_supports_posix_fd(device: int) -> bool:
    _cuda._init()
    attr = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED
    err, val = driver.cuDeviceGetAttribute(attr, device)
    return err == driver.CUresult.CUDA_SUCCESS and int(val) == 1


def cu_map_handle(handle: object, size: int, device: int) -> int:
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
    _cuda._init()
    prop = _alloc_prop(device, kind)
    err, gran = driver.cuMemGetAllocationGranularity(
        prop,
        driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
    )
    _check(err, "cuMemGetAllocationGranularity")
    granularity = int(gran)
    if cu_device_supports_multicast(device):
        granularity = max(
            granularity, cu_multicast_granularity(_MC_GRANULARITY_DEVICES, kind)
        )
    size = _round_up(min_size, granularity)
    err, handle = driver.cuMemCreate(size, prop, 0)
    _check(err, "cuMemCreate")
    try:
        base = cu_map_handle(handle, size, device)
    except Exception as e:
        driver.cuMemRelease(handle)
        raise e
    return base, size, handle


def cu_mem_export(
    handle: object, kind: IpcHandleType = IpcHandleType.FABRIC
) -> bytes | int:
    err, sh = driver.cuMemExportToShareableHandle(handle, _driver_handle_type(kind), 0)
    _check(err, "cuMemExportToShareableHandle")
    if kind is IpcHandleType.FABRIC:
        return bytes(sh.data)
    return int(sh)


def cu_mem_import(
    raw: bytes | int, kind: IpcHandleType = IpcHandleType.FABRIC
) -> object:
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
    handle = cu_mem_import(raw, kind)
    try:
        return cu_map_handle(handle, size, device), handle
    except Exception as e:
        cu_mem_release(handle)
        raise e


def cu_mem_unmap(base: int, size: int) -> bool:
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
    unmapped = cu_mem_unmap(base, size)
    released = cu_mem_release(handle)
    return unmapped and released


_MULTICAST_ATTR: object = (
    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
)


@functools.cache
def cu_device_supports_multicast(device: int) -> bool:
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
    _cuda._init()
    prop = _multicast_prop(num_devices, size, kind)
    err, mc = driver.cuMulticastCreate(prop)
    _check(err, "cuMulticastCreate")
    return mc


def cu_multicast_add_device(mc: object, device: int) -> None:
    (err,) = driver.cuMulticastAddDevice(mc, device)
    _check(err, "cuMulticastAddDevice")


def cu_multicast_bind_mem(mc: object, mem_handle: object, size: int) -> None:
    (err,) = driver.cuMulticastBindMem(mc, 0, mem_handle, 0, size, 0)
    _check(err, "cuMulticastBindMem")


def cu_multicast_unbind(mc: object, device: int, size: int) -> None:
    (err,) = driver.cuMulticastUnbind(mc, device, 0, size)
    _check(err, "cuMulticastUnbind")


@dataclass
class _CudaMapping:
    base: int
    size: int
    handle: object
    device: int


@dataclass
class _CudaMulticast:
    handle: object
    addr: int
    size: int
    device: int


@dataclass
class _CudaPeerState:
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
        if _cuda.should_skip_cuda_cleanup(m.device):
            return CleanupStatus.DEFERRED
        with torch.cuda.device(m.device):
            if not cu_vmm_free(m.base, m.size, m.handle):
                return CleanupStatus.FAILED
        return CleanupStatus.CLEANED

    def allocation_base(self, ptr: int) -> int:
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
        local_error: Exception | None = None
        exported: int | None = None
        try:
            if dev_idx != mine.device:
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
        handles, fds = _share_handles(
            exported, gathered, pg, rank, world_size, transport
        )
        signal_pads: list[int] = []
        imported: list[_CudaMapping] = []
        try:
            for peer_rank in range(world_size):
                if peer_rank == rank:
                    signal_pads.append(mine.base)
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
            for m in imported:
                cu_vmm_free(m.base, m.size, m.handle)
            raise e
        finally:
            _ipc_fd.close_fds(fds)

    @staticmethod
    def _teardown_multicast(
        mc_handle: object, bound: bool, mc_addr: int, size: int, device: int
    ) -> bool:
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
        if world_size < 2 or not all(g["multicast"] for g in gathered):
            return None, 0

        size = mine.size
        mc_handle: object = None
        bound = False
        mc_addr = 0

        def all_agree(ok: bool) -> bool:
            return _all_agree(pg, world_size, ok)

        try:
            mc_exported: bytes | int | None = None
            if rank == 0:
                try:
                    mc_handle = cu_multicast_create(size, world_size, transport)
                    mc_exported = cu_mem_export(mc_handle, transport)
                except Exception:
                    logger.warning("multicast create/export failed", exc_info=True)
            if not all_agree(rank != 0 or mc_exported is not None):
                self._teardown_multicast(mc_handle, bound, mc_addr, size, dev_idx)
                return None, 0

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
        owner = mc.device if mc is not None else state.imported[0].device
        if _cuda.should_skip_cuda_cleanup(owner):
            return CleanupStatus.DEFERRED
        ok = True
        with torch.cuda.device(owner):
            if mc is not None:
                ok = self._teardown_multicast(
                    mc.handle, True, mc.addr, mc.size, mc.device
                )
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
        _ipc_fd.close_fds(fds)
        raise RuntimeError("fd exchange lost the multicast handle from rank 0")
    return fds[0], fds
