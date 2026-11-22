# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


class PoolBackend(Protocol):
    def alloc(self, size: int, device: int, stream: int) -> int: ...

    def free(self, buffer_ptr: int, size: int, device: int, stream: int) -> None: ...

    def should_skip_cleanup(self, device: int) -> bool: ...

    def release_peer_resources(self, device: int) -> int: ...

    def has_unreleased_resources(self, device: int) -> bool: ...

    def has_pending_releases(self, device: int) -> bool: ...

    def release_pending(self, device: int) -> int: ...


_backend_resolver: Callable[[], PoolBackend] | None = None


def set_backend_resolver(resolver: Callable[[], PoolBackend]) -> None:
    global _backend_resolver
    _backend_resolver = resolver


def get_backend() -> PoolBackend:
    if _backend_resolver is None:
        raise RuntimeError("no symmetric-memory backend has been registered")
    return _backend_resolver()


def alloc(size: int, device: int, stream: int) -> int:
    return get_backend().alloc(size, device, stream)


def free(ptr: int, size: int, device: int, stream: int) -> None:
    get_backend().free(ptr, size, device, stream)
