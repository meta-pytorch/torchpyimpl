# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import socket
import time
from contextlib import closing
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torchpyimpl.distributed.symmetric_memory as symm
from torchpyimpl.distributed.symmetric_memory import _api


if TYPE_CHECKING:
    from collections.abc import Callable


_OPS = ("barrier", "barrier_cas", "put_wait")


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _quantiles(xs: list[float]) -> tuple[float, float, float]:
    xs = sorted(xs)
    return tuple(xs[int(q * (len(xs) - 1))] for q in (0.2, 0.5, 0.8))  # type: ignore[return-value]


def _op(hdl: symm.SymmMemHandle, name: str) -> Callable[[], None]:
    nxt, prv = (hdl.rank + 1) % hdl.world_size, (hdl.rank - 1) % hdl.world_size
    mc = hdl.multicast_signal_pad_ptr

    def barrier() -> None:
        hdl.barrier()

    def barrier_cas() -> None:
        hdl.multicast_signal_pad_ptr = 0
        try:
            hdl.barrier()
        finally:
            hdl.multicast_signal_pad_ptr = mc

    def put_wait() -> None:
        hdl.put_signal(nxt)
        hdl.wait_signal(prv)

    return {"barrier": barrier, "barrier_cas": barrier_cas, "put_wait": put_wait}[name]


def _time(
    op: Callable[[], None], streams: list[torch.cuda.Stream], batch: int, iters: int
) -> dict[str, tuple[float, float, float]]:
    def run() -> float:
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter_ns()
        for i in range(batch):
            with torch.cuda.stream(streams[i % len(streams)]):
                op()
        return (time.perf_counter_ns() - t0) / 1e3 / batch

    for _ in range(3):
        run()
    issue, total = [], []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        issue.append(run())
        torch.cuda.synchronize()
        total.append((time.perf_counter_ns() - t0) / 1e3 / batch)
    return {"issue_us": _quantiles(issue), "total_us": _quantiles(total)}


def _worker(rank: int, world_size: int, port: int, args: argparse.Namespace) -> None:
    logging.getLogger("torchpyimpl").setLevel(logging.WARNING)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
    symm.set_backend(args.backend)
    hdl = symm.rendezvous(symm.empty(64, device=device), dist.group.WORLD)
    one, two = [torch.cuda.Stream()], [torch.cuda.Stream(), torch.cuda.Stream()]
    for s in one + two:
        s.wait_stream(torch.cuda.current_stream())
    guarded = _api.group_stream_guard
    rows = []
    for name in _OPS:
        if name != "put_wait" and world_size < 2:
            continue
        op = _op(hdl, name)
        for guard, label, streams in (
            (True, "one", one),
            (False, "one", one),
            (True, "two", two),
        ):
            _api.group_stream_guard = (  # type: ignore[assignment]
                guarded if guard else lambda _hdl: contextlib.nullcontext()
            )
            try:
                q = _time(op, streams, args.batch, args.iters)
            finally:
                _api.group_stream_guard = guarded
            rows.append((name, guard, label, q))
    if rank == 0:
        _print(rows, args, world_size, hdl.multicast_signal_pad_ptr != 0)
    dist.barrier()
    dist.destroy_process_group()


def _print(rows: list[Any], args: argparse.Namespace, world: int, mc: bool) -> None:
    print(
        f"backend={args.backend} world_size={world} multicast={mc} "
        f"batch={args.batch} iters={args.iters}  (us per op, p20/p50/p80, rank 0)"
    )
    print(f"{'op':<12}{'guard':<7}{'streams':<9}{'issue':>24}{'total':>24}")
    for name, guard, label, q in rows:
        issue = "/".join(f"{x:.2f}" for x in q["issue_us"])
        total = "/".join(f"{x:.2f}" for x in q["total_us"])
        print(
            f"{name:<12}{'on' if guard else 'off':<7}{label:<9}{issue:>24}{total:>24}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("CUDA", "NCCL"), default="CUDA")
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    world_size = args.world_size or min(torch.cuda.device_count(), 8)
    mp.spawn(_worker, args=(world_size, _free_port(), args), nprocs=world_size)


if __name__ == "__main__":
    main()
