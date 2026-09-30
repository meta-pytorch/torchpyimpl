# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    gpus = torch.cuda.device_count()
    skip_gpu = pytest.mark.skip(reason="requires a CUDA GPU")
    skip_multi_gpu = pytest.mark.skip(reason="requires two or more CUDA GPUs")
    for item in items:
        if "gpu" in item.keywords and gpus < 1:
            item.add_marker(skip_gpu)
        if "multi_gpu" in item.keywords and gpus < 2:
            item.add_marker(skip_multi_gpu)

    pattern = os.environ.get("DISABLED_TESTS")
    if not pattern:
        return

    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise pytest.UsageError(f"Invalid DISABLED_TESTS regex {pattern!r}: {e}")

    skip_marker = pytest.mark.skip(reason=f"disabled via DISABLED_TESTS={pattern!r}")
    for item in items:
        if regex.search(item.nodeid):
            item.add_marker(skip_marker)
