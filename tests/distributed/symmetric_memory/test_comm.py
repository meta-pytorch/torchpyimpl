# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import socket
import tempfile
from contextlib import closing
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _dist_worker import dist_worker

from torchpyimpl.distributed.symmetric_memory._comm import (
    _synthetic_group_id,
    clear_comm_ptr_cache,
    get_group_info,
    get_nccl_comm_ptr,
)


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dist_worker
def _comm_worker(rank: int, world_size: int, result_path: str) -> None:
    device = torch.device(f"cuda:{rank}")
    pg = dist.new_group(ranks=list(range(world_size)), backend="nccl")

    p1 = get_nccl_comm_ptr(pg, device)
    p2 = get_nccl_comm_ptr(pg, device)
    clear_comm_ptr_cache()
    p3 = get_nccl_comm_ptr(pg, device)

    synthetic = get_group_info(pg, device, require_comm=False).comm_ptr
    p4 = get_nccl_comm_ptr(pg, device)

    dev_index = device.index if device.index is not None else 0
    other_dev = _synthetic_group_id(pg, dev_index + 1)

    report: dict[str, Any] = {
        "rank": rank,
        "nonzero": p1 != 0,
        "memoized": p2 == p1,
        "reresolved_same": p3 == p1,
        "synthetic_negative": synthetic < 0,
        "real_after_synthetic": p4 == p1,
        "synthetic_per_device": other_dev != synthetic,
    }
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, report, group=pg)
    if rank == 0:
        with open(result_path, "w") as f:
            json.dump({"reports": gathered}, f)
    dist.destroy_process_group(pg)


@pytest.mark.multi_gpu
def test_get_nccl_comm_ptr_memoized_2gpu() -> None:
    world_size = min(torch.cuda.device_count(), 8)
    if world_size < 2:
        pytest.skip("need >= 2 CUDA devices")

    master_port = _free_port()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        result_path = tmp.name

    mp.spawn(
        _comm_worker,
        args=(world_size, master_port, result_path),
        nprocs=world_size,
        join=True,
    )

    with open(result_path) as f:
        reports = json.load(f)["reports"]

    assert len(reports) == world_size
    for r in reports:
        rank = r["rank"]
        assert r["nonzero"], f"rank {rank}: borrowed comm ptr is 0"
        assert r["memoized"], f"rank {rank}: 2nd get_nccl_comm_ptr not memoized"
        assert r["reresolved_same"], (
            f"rank {rank}: re-resolve after clear_comm_ptr_cache differs"
        )
        assert r["synthetic_negative"], (
            f"rank {rank}: synthetic id is not negative, so it can collide with "
            "a real communicator address"
        )
        assert r["real_after_synthetic"], (
            f"rank {rank}: get_nccl_comm_ptr returned the synthetic id cached by "
            "a require_comm=False lookup instead of the ncclComm_t"
        )
        assert r["synthetic_per_device"], (
            f"rank {rank}: one group on two devices collapsed to one synthetic id"
        )
