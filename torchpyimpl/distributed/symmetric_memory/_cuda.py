# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA driver wrappers (``cuda.bindings.driver``).

``cu_memset_d8`` does not go through the CUDA caching allocator, so it is safe
inside MemPool allocator callbacks.
"""

from __future__ import annotations

import functools
import logging
import sys

import torch
from cuda.bindings import driver


logger: logging.Logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _init() -> None:
    (err,) = driver.cuInit(0)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {err}")


def cu_mem_get_address_range(ptr: int) -> int:
    _init()
    err, base, _size = driver.cuMemGetAddressRange(ptr)
    if err == driver.CUresult.CUDA_ERROR_NOT_FOUND:
        logger.debug("no device allocation covers 0x%x (%s)", ptr, err)
        return 0
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemGetAddressRange failed at 0x{ptr:x}: {err}")
    return int(base)


def cu_stream_write_value32(ptr: int, value: int, stream: int) -> None:
    _init()
    (err,) = driver.cuStreamWriteValue32(
        stream,
        ptr,
        value & 0xFFFFFFFF,
        driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
    )
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuStreamWriteValue32 failed at 0x{ptr:x}: {err}")


def cu_memset_d8(ptr: int, value: int, n: int) -> None:
    if n < 0:
        raise ValueError(f"cu_memset_d8 length must be non-negative, got {n}")
    if n == 0:
        return
    _init()
    (err,) = driver.cuMemsetD8(ptr, value & 0xFF, n)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemsetD8 failed at 0x{ptr:x}: {err}")
    cu_ctx_synchronize()


def cu_ctx_synchronize() -> None:
    _init()
    (err,) = driver.cuCtxSynchronize()
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxSynchronize failed: {err}")


def should_skip_cuda_cleanup(device: int) -> bool:
    if sys.is_finalizing():
        return True
    try:
        with torch.cuda.device(device):
            cu_ctx_synchronize()
        return False
    except RuntimeError as e:
        logger.warning(
            "skipping symmetric-memory cleanup on cuda:%d after CUDA error: %s",
            device,
            e,
        )
        return True
