# Copyright (c) Meta Platforms, Inc. and affiliates.

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
