# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the public symmetric memory API."""

from __future__ import annotations

import pytest
import torch

import torchpyimpl.distributed.symmetric_memory as symm
import torchpyimpl.distributed.symmetric_memory._symm_mem as sm


def test_set_backend_validation() -> None:
    assert sm.get_active_backend() == "CUDA"
    symm.set_backend("CUDA")
    with pytest.raises(ValueError):
        symm.set_backend("NVSHMEM")
    assert sm.get_active_backend() == "CUDA"


def test_default_signal_pad_size_matches_torch() -> None:
    # torch's default: symm_max_nblocks(32) * max_p2p_domain(72) * 4.
    assert symm.get_signal_pad_size() == 32 * 72 * 4


def test_set_signal_pad_size_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        symm.set_signal_pad_size(0)
    with pytest.raises(ValueError):
        symm.set_signal_pad_size(-16)


def test_empty_rejects_non_cuda_device() -> None:
    with pytest.raises(ValueError, match="CUDA device"):
        symm.empty(4, device=torch.device("cpu"))
