# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""Pure-Python implementations of PyTorch GPU and distributed subsystems."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("torchpyimpl")
except PackageNotFoundError:
    __version__ = "unknown"
