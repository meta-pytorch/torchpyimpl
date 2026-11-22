# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-Python implementations of PyTorch GPU and distributed subsystems."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__: str = version("torchpyimpl")
except PackageNotFoundError:
    __version__ = "unknown"
