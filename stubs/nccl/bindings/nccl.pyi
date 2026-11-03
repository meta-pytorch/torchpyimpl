# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# nccl.bindings.nccl is a compiled Cython module that ships no type stubs.

from typing import Any

def __getattr__(name: str) -> Any: ...
