# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest

import torchpyimpl


SUBPACKAGES = [
    "torchpyimpl.distributed",
    "torchpyimpl.distributed.symmetric_memory",
]


def test_version() -> None:
    assert torchpyimpl.__version__ != "unknown"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_import(name: str) -> None:
    importlib.import_module(name)
