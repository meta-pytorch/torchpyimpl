# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Skip symmetric memory tests on hosts that lack the hardware they need.

The probes try the operation rather than read device attributes: a GPU can
report fabric handle support on a host with no IMEX channel, where
``cuMemCreate`` then fails with ``CUDA_ERROR_NOT_PERMITTED``.
"""

from __future__ import annotations

import functools

import pytest
import torch
from cuda.bindings import driver


_OK = driver.CUresult.CUDA_SUCCESS
_FABRIC = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
_MAX_RANKS = 8  # multi-GPU tests use min(device_count, 8) ranks


@functools.cache
def _peer_access() -> bool:
    n = min(torch.cuda.device_count(), _MAX_RANKS)
    return n >= 2 and all(
        torch.cuda.can_device_access_peer(i, j)
        for i in range(n)
        for j in range(n)
        if i != j
    )


@functools.cache
def _sm90() -> bool:
    return torch.cuda.get_device_capability(0) >= (9, 0)


@functools.cache
def _fabric_handles() -> bool:
    # Create, export, and import a fabric handle on GPU 0, as PyTorch's
    # isFabricSupported does.
    (err,) = driver.cuInit(0)
    if err != _OK:
        return False
    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = 0
    prop.requestedHandleTypes = _FABRIC
    err, gran = driver.cuMemGetAllocationGranularity(
        prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    )
    if err != _OK:
        return False
    err, handle = driver.cuMemCreate(int(gran), prop, 0)
    if err != _OK:
        return False
    try:
        err, shared = driver.cuMemExportToShareableHandle(handle, _FABRIC, 0)
        if err != _OK:
            return False
        err, imported = driver.cuMemImportFromShareableHandle(
            bytes(shared.data), _FABRIC
        )
        if err != _OK:
            return False
        driver.cuMemRelease(imported)
        return True
    finally:
        driver.cuMemRelease(handle)


_PROBES = {
    "peer_access": (_peer_access, "requires peer access between the GPUs"),
    "fabric": (_fabric_handles, "requires usable CUDA fabric handles"),
    "sm90": (_sm90, "requires compute capability 9.0+"),
}


def pytest_collection_modifyitems(config, items):
    if torch.cuda.device_count() < 1:
        return  # The root conftest skips the GPU tests.
    for item in items:
        for name, (probe, reason) in _PROBES.items():
            if item.get_closest_marker(name) is not None and not probe():
                item.add_marker(pytest.mark.skip(reason=reason))
