# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import socket
from contextlib import closing
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import pathlib

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from torchpyimpl.distributed.symmetric_memory.backends import _ipc_fd


_WORLD_SIZE = 4


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _read(fd: int) -> str:
    return os.pread(fd, 64, 0).decode()


def _worker(rank: int, world_size: int, port: int, tmp: str, result_path: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        pg = dist.group.WORLD
        path = os.path.join(tmp, f"rank{rank}")
        with open(path, "w") as f:
            f.write(f"rank{rank}")
        fd = os.open(path, os.O_RDONLY)

        fds = _ipc_fd.all_gather_fds(fd, pg)
        contents = [_read(f) if f is not None else None for f in fds]
        own_returned = fds[rank] == fd
        _ipc_fd.close_fds([f for i, f in enumerate(fds) if i != rank])

        root = _ipc_fd.all_gather_fds(fd if rank == 0 else None, pg)
        root_content = _read(root[0]) if root[0] is not None else None
        others_none = all(f is None for i, f in enumerate(root) if i not in (0,))
        if rank != 0 and root[0] is not None:
            os.close(root[0])
        os.close(fd)

        report: dict[str, Any] = {
            "rank": rank,
            "contents": contents,
            "own_returned": own_returned,
            "root_content": root_content,
            "others_none": others_none,
            "leftover_sockets": [
                p
                for p in os.listdir(os.path.dirname(_ipc_fd._socket_path()))
                if p.startswith(f"torchpyimpl-{os.getpid()}-")
            ],
        }
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, report)
        if rank == 0:
            with open(result_path, "w") as f:
                json.dump(gathered, f)
    finally:
        dist.destroy_process_group()


def test_all_gather_fds(tmp_path: pathlib.Path) -> None:
    result_path = str(tmp_path / "reports.json")
    mp.spawn(
        _worker,
        args=(_WORLD_SIZE, _free_port(), str(tmp_path), result_path),
        nprocs=_WORLD_SIZE,
        join=True,
    )
    with open(result_path) as f:
        reports = json.load(f)

    expected = [f"rank{r}" for r in range(_WORLD_SIZE)]
    for r in reports:
        rank = r["rank"]
        assert r["contents"] == expected, f"rank {rank}: {r['contents']}"
        assert r["own_returned"], f"rank {rank}: local fd not returned at its index"
        assert r["root_content"] == "rank0", f"rank {rank}: {r['root_content']}"
        assert r["others_none"], f"rank {rank}: non-root slots were not None"
        assert not r["leftover_sockets"], f"rank {rank}: {r['leftover_sockets']}"


def _failure_worker(rank: int, world_size: int, port: int, result_path: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        if rank == 1:
            recv_fd = _ipc_fd._recv_fd
            calls = [0]

            def recv_then_fail(sock: socket.socket) -> int | None:
                calls[0] += 1
                fd = recv_fd(sock)
                if calls[0] == world_size - 1:
                    raise RuntimeError("injected receive failure")
                return fd

            _ipc_fd._recv_fd = recv_then_fail
        error = ""
        try:
            _ipc_fd.all_gather_fds(os.open(os.devnull, os.O_RDONLY), dist.group.WORLD)
        except RuntimeError as e:
            error = str(e)
        errors: list[Any] = [None] * world_size
        dist.all_gather_object(errors, error)
        if rank == 0:
            with open(result_path, "w") as f:
                json.dump(errors, f)
    finally:
        dist.destroy_process_group()


def test_all_gather_fds_raises_on_every_rank_when_one_rank_fails(
    tmp_path: pathlib.Path,
) -> None:
    result_path = str(tmp_path / "errors.json")
    mp.spawn(
        _failure_worker,
        args=(_WORLD_SIZE, _free_port(), result_path),
        nprocs=_WORLD_SIZE,
        join=True,
    )
    with open(result_path) as f:
        errors = json.load(f)
    assert "injected" in errors[1], errors
    assert all("rank(s) [1] failed" in e for i, e in enumerate(errors) if i != 1), (
        errors
    )


def test_single_rank_returns_local_fd() -> None:
    assert _ipc_fd.all_gather_fds(7, None) == [7]


@pytest.mark.parametrize("fds", [[], [None], [None, None]])
def test_close_fds_ignores_none(fds: list[int | None]) -> None:
    _ipc_fd.close_fds(fds)
