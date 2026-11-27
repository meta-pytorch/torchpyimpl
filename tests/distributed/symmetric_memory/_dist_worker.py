# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import os
from typing import Any, overload, TYPE_CHECKING

import torch
import torch.distributed as dist


if TYPE_CHECKING:
    from collections.abc import Callable

    Worker = Callable[..., None]


@overload
def dist_worker(fn: Worker) -> Worker: ...


@overload
def dist_worker(*, backend: str) -> Callable[[Worker], Worker]: ...


def dist_worker(
    fn: Worker | None = None, *, backend: str = "nccl"
) -> Worker | Callable[[Worker], Worker]:
    def wrap(fn: Worker) -> Worker:
        @functools.wraps(fn)
        def run(rank: int, world_size: int, port: int, *args: Any) -> None:
            os.environ.update(
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT=str(port),
                RANK=str(rank),
                WORLD_SIZE=str(world_size),
                LOCAL_RANK=str(rank),
            )
            if backend == "nccl":
                device = torch.device(f"cuda:{rank}")
                torch.cuda.set_device(device)
                dist.init_process_group(
                    backend, rank=rank, world_size=world_size, device_id=device
                )
            else:
                dist.init_process_group(backend, rank=rank, world_size=world_size)
            try:
                fn(rank, world_size, *args)
            finally:
                dist.destroy_process_group()

        return run

    return wrap if fn is None else wrap(fn)
