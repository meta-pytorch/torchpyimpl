# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import socket
import stat
import tempfile
from contextlib import closing
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import pathlib

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from _dist_worker import dist_worker

from torchpyimpl.distributed.symmetric_memory.backends import _ipc_fd


_WORLD_SIZE = 4


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _read(fd: int) -> str:
    return os.pread(fd, 64, 0).decode()


@dist_worker(backend="gloo")
def _worker(rank: int, world_size: int, tmp: str, result_path: str) -> None:
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
            for p in os.listdir(tempfile.gettempdir())
            if p.startswith(f"torchpyimpl-{os.getpid()}-")
        ],
    }
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, report)
    if rank == 0:
        with open(result_path, "w") as f:
            json.dump(gathered, f)


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


@dist_worker(backend="gloo")
def _failure_worker(rank: int, world_size: int, result_path: str) -> None:
    if rank == 1:
        recv_fd = _ipc_fd._recv_fd
        calls = [0]

        def recv_then_fail(sock: socket.socket, src_path: str) -> int | None:
            calls[0] += 1
            fd = recv_fd(sock, src_path)
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


def test_socket_dir_is_private_under_permissive_umask() -> None:
    old_umask = os.umask(0)
    try:
        sock, sock_dir = _ipc_fd._bind_socket()
    finally:
        os.umask(old_umask)
    try:
        st = os.stat(sock_dir)
        assert stat.S_IMODE(st.st_mode) == 0o700
        assert st.st_uid == os.getuid()
        assert os.path.dirname(sock.getsockname()) == sock_dir
    finally:
        sock.close()
        _ipc_fd._remove_dir(sock_dir)
    assert not os.path.exists(sock_dir)


def _open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_recv_fd_rejects_unexpected_sender_and_closes_its_fd() -> None:
    bound = [_ipc_fd._bind_socket() for _ in range(3)]
    receiver, peer, intruder = (s for s, _ in bound)
    try:
        fd = os.open(os.devnull, os.O_RDONLY)
        _ipc_fd._send_fd(intruder, receiver.getsockname(), fd)
        os.close(fd)
        before = _open_fd_count()
        with pytest.raises(RuntimeError, match="expected the previous rank"):
            _ipc_fd._recv_fd(receiver, peer.getsockname())
        assert _open_fd_count() == before

        _ipc_fd._send_fd(peer, receiver.getsockname(), None)
        assert _ipc_fd._recv_fd(receiver, peer.getsockname()) is None
    finally:
        for sock, sock_dir in bound:
            sock.close()
            _ipc_fd._remove_dir(sock_dir)


def test_recv_fd_rejects_unexpected_payload() -> None:
    bound = [_ipc_fd._bind_socket() for _ in range(2)]
    receiver, peer = (s for s, _ in bound)
    try:
        peer.sendto(b"xyz", receiver.getsockname())
        with pytest.raises(RuntimeError, match="unexpected payload"):
            _ipc_fd._recv_fd(receiver, peer.getsockname())
    finally:
        for sock, sock_dir in bound:
            sock.close()
            _ipc_fd._remove_dir(sock_dir)
