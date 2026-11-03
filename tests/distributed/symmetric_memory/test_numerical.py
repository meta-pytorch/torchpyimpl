# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton one-shot all-reduce over symmetric memory, checked bit-exactly."""

from __future__ import annotations

import json
import os
import socket
from contextlib import closing
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import pathlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl

import torchpyimpl.distributed.symmetric_memory as symm


_SEED = 42
_SIZES = [1024, 16384, 9973]  # power-of-2, larger, and a prime (masked tail)


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gen_payload(rank: int, n: int, device: torch.device) -> torch.Tensor:
    """Deterministic per-rank fp32 payload; regenerable on any rank."""
    gen = torch.Generator(device=device).manual_seed(_SEED + rank)
    return torch.empty(n, dtype=torch.float32, device=device).uniform_(
        -1.0, 1.0, generator=gen
    )


@triton.jit
def _one_shot_all_reduce_kernel(
    peer_ptrs,  # *i64: world_size entries, each a peer's fp32 buffer base
    out_ptr,  # *f32
    n_elements,
    world_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for r in tl.static_range(world_size):
        base = tl.load(peer_ptrs + r).to(tl.pointer_type(tl.float32))
        acc += tl.load(base + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


def _one_shot_all_reduce(
    hdl: symm.SymmMemHandle, n: int, world_size: int, device: torch.device
) -> torch.Tensor:
    peer = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    out = torch.empty(n, dtype=torch.float32, device=device)
    block = 1024
    grid = (triton.cdiv(n, block),)
    _one_shot_all_reduce_kernel[grid](peer, out, n, world_size=world_size, BLOCK=block)
    return out


def _worker(
    rank: int, world_size: int, master_port: int, result_path: str, backend: str
) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device
    )
    try:
        symm.set_backend(backend)
        pg = dist.new_group(ranks=list(range(world_size)))

        per_size: dict[str, Any] = {}
        for n in _SIZES:
            buf = symm.empty(n, dtype=torch.float32, device=device)
            buf.copy_(_gen_payload(rank, n, device))
            torch.cuda.synchronize(device)
            hdl = symm.rendezvous(buf, pg)

            # Every rank must finish its fill before any rank reads peers.
            hdl.barrier()
            out = _one_shot_all_reduce(hdl, n, world_size, device)
            torch.cuda.synchronize(device)

            expected = torch.zeros(n, dtype=torch.float32, device=device)
            for r in range(world_size):
                expected += _gen_payload(r, n, device)

            per_size[str(n)] = {
                "max_abs_err": (out - expected).abs().max().item(),
                "out_first4": out[:4].tolist(),
                "exp_first4": expected[:4].tolist(),
            }
            # Nobody may reuse or free the buffer while a peer is still reading.
            hdl.barrier()
            torch.cuda.synchronize(device)

        symm.free_pool(device)

        report: dict[str, Any] = {"rank": rank, "per_size": per_size}
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, report, group=pg)
        if rank == 0:
            with open(result_path, "w") as f:
                json.dump({"reports": gathered}, f)
        dist.destroy_process_group(pg)
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.peer_access
@pytest.mark.parametrize("backend", ["CUDA", "NCCL"])
def test_triton_oneshot_allreduce_numerical(
    backend: str, tmp_path: pathlib.Path
) -> None:
    world_size = min(torch.cuda.device_count(), 8)
    result_path = str(tmp_path / "reports.json")
    mp.spawn(
        _worker,
        args=(world_size, _free_port(), result_path, backend),
        nprocs=world_size,
        join=True,
    )

    with open(result_path) as f:
        reports = json.load(f)["reports"]

    assert len(reports) == world_size
    for r in reports:
        rank = r["rank"]
        for n, res in r["per_size"].items():
            # The kernel sums in rank order and so does the reference, so fp32
            # must agree bit-exactly.
            assert res["max_abs_err"] == 0.0, (
                f"rank {rank}, size {n}: all-reduce diverges by "
                f"{res['max_abs_err']} (out={res['out_first4']}, "
                f"exp={res['exp_first4']})"
            )
