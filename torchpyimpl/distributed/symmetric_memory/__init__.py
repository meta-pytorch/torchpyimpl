# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Symmetric memory, modeled on torch.distributed._symmetric_memory.

The accessors are module functions that take a handle. The signal-pad
operations (barrier, put_signal, wait_signal) are handle methods, as in torch.
"""

from torchpyimpl.distributed.symmetric_memory._api import (
    empty,
    free_pool,
    get_buffer,
    get_remote_tensor,
    get_signal_pad,
    get_signal_pad_size,
    has_multicast_support,
    rendezvous,
    set_backend,
    set_signal_pad_size,
)
from torchpyimpl.distributed.symmetric_memory._symm_mem import SymmMemHandle


__all__ = [
    "SymmMemHandle",
    "empty",
    "free_pool",
    "get_buffer",
    "get_remote_tensor",
    "get_signal_pad",
    "get_signal_pad_size",
    "has_multicast_support",
    "rendezvous",
    "set_backend",
    "set_signal_pad_size",
]
