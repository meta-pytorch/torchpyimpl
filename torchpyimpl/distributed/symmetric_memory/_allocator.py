# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Private: a ``torch._C._cuda_CUDAAllocator`` backed by the registered
symmetric-memory backend, built via ``ctypes``.

``torch.cuda.MemPool`` needs a ``torch._C._cuda_CUDAAllocator``: a pair of C
function pointers with the signatures::

    void* alloc(size_t size, int device, cudaStream_t stream);
    void  free (void* ptr, size_t size, int device, cudaStream_t stream);

We build the two callables with ``ctypes.CFUNCTYPE`` and route them to the
:class:`PoolBackend` returned by the resolver registered with
:func:`set_backend_resolver`. The backend does the signal-pad-first layout,
physical allocation, and record-keeping, so every block the CUDA caching
allocator requests from the pool is symmetric memory — mirroring torch's
``cuda_symm_alloc -> SymmetricMemoryAllocator``.

Caveats of the ctypes route (fine for one-process-per-rank SPMD):

* The callback objects must outlive the pool, so :func:`_get_allocator` pins
  every pair it builds in ``_pinned``, which is only ever appended to.
* A Python exception cannot propagate back into C++, so the callbacks catch
  everything: alloc returns a null pointer (reads as OOM to the caching
  allocator), free logs and swallows.
* The alloc path stays below the torch caching allocator, so it is safe to run
  while that allocator holds its lock.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any, Callable, Protocol

import torch

logger: logging.Logger = logging.getLogger(__name__)

# CUDAPluggableAllocator ABI: stream is passed as cudaStream_t, taken here as a
# void* we never dereference.
_ALLOC_FN: type[ctypes._FuncPointer] = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
)
_FREE_FN: type[ctypes._FuncPointer] = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
)


class PoolBackend(Protocol):
    """What the pluggable allocator and the symmetric ``MemPool`` need from a
    symmetric-memory backend."""

    def alloc(self, size: int, device: int) -> int: ...

    def free(self, ptr: int) -> None: ...

    def should_skip_cleanup(self, device: int) -> bool: ...

    def deregister_windows(self, device: int) -> int: ...

    def has_unreleased_resources(self, device: int) -> bool: ...

    def has_pending_releases(self, device: int) -> bool: ...

    def release_pending(self, device: int) -> int: ...


_backend_resolver: Callable[[], PoolBackend] | None = None


def set_backend_resolver(resolver: Callable[[], PoolBackend]) -> None:
    """Register the callable that returns the active :class:`PoolBackend`.

    It is called on every alloc/free, so switching the active backend does not
    require rebuilding the C function pointers."""
    global _backend_resolver
    _backend_resolver = resolver


def get_backend() -> PoolBackend:
    if _backend_resolver is None:
        raise RuntimeError("no symmetric-memory backend has been registered")
    return _backend_resolver()


_allocator: torch._C._cuda_CUDAAllocator | None = None
# Every callback pair ever built, kept alive for the process lifetime. A list
# rather than two slots: a pair is only ever appended, never replaced, so the
# raw addresses inside a live MemPool can never be freed underneath it.
_pinned: list[tuple[Any, Any]] = []


def _get_allocator() -> torch._C._cuda_CUDAAllocator:
    """The shared CUDA allocator backed by the registered symmetric-memory backend.

    The callbacks resolve :func:`get_backend` **per call**, so the one
    pluggable allocator (and its MemPool) routes to whichever backend is active
    without rebuilding the C function pointers.

    Cached, so repeat calls are a null check rather than a rebuild. Private and
    unsynchronized: ``_mempool.get_mem_pool`` is the only caller and already holds
    its own lock. Should that ever stop being true, an unserialized double build
    is still safe — ``_pinned`` is append-only, so both pairs stay referenced —
    it would only waste one allocator."""
    global _allocator
    if _allocator is not None:
        return _allocator

    def _alloc(size: int, device: int, stream: int | None) -> int:
        try:
            return get_backend().alloc(size, device)
        except Exception:
            # Can't raise across the C ABI; a null return reads as OOM.
            logger.exception(f"symmetric alloc failed for size {size}")
            return 0

    def _free(ptr: int, size: int, device: int, stream: int | None) -> None:
        try:
            get_backend().free(ptr)
        except Exception:
            logger.exception(f"symmetric free failed for ptr 0x{ptr:x}")

    alloc_cb = _ALLOC_FN(_alloc)
    free_cb = _FREE_FN(_free)
    _pinned.append((alloc_cb, free_cb))
    alloc_addr = ctypes.cast(alloc_cb, ctypes.c_void_p).value
    free_addr = ctypes.cast(free_cb, ctypes.c_void_p).value
    if alloc_addr is None or free_addr is None:
        raise RuntimeError("could not take the address of the allocator callbacks")
    logger.info("built symmetric CUDA allocator")
    _allocator = torch._C._cuda_customAllocator(alloc_addr, free_addr)
    return _allocator
