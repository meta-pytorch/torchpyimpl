# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import array
import logging
import os
import shutil
import socket
import tempfile

import torch.distributed as dist


logger: logging.Logger = logging.getLogger(__name__)

_RECV_TIMEOUT_S: float = 300.0
_PAYLOAD: bytes = b"fd"
_SOCKET_NAME: str = "sock"


def _bind_socket() -> tuple[socket.socket, str]:
    sock_dir = tempfile.mkdtemp(prefix=f"torchpyimpl-{os.getpid()}-")
    sock: socket.socket | None = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(os.path.join(sock_dir, _SOCKET_NAME))
        sock.settimeout(_RECV_TIMEOUT_S)
    except BaseException:
        if sock is not None:
            sock.close()
        _remove_dir(sock_dir)
        raise
    return sock, sock_dir


def _remove_dir(sock_dir: str) -> None:
    try:
        shutil.rmtree(sock_dir)
    except OSError:
        logger.debug("could not remove ipc socket dir %s", sock_dir)


def all_gather_fds(
    local_fd: int | None,
    pg: dist.ProcessGroup | None,
) -> list[int | None]:
    if pg is None:
        return [local_fd]
    rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return [local_fd]

    sock: socket.socket | None = None
    sock_dir = ""
    setup_error: Exception | None = None
    try:
        sock, sock_dir = _bind_socket()
    except Exception as e:
        setup_error = e

    fds: list[int | None] = [None for _ in range(world_size)]
    try:
        paths: list[str] = [""] * world_size
        dist.all_gather_object(
            paths, sock.getsockname() if sock is not None else "", group=pg
        )
        if not all(paths):
            if setup_error is not None:
                raise setup_error
            missing = [i for i, q in enumerate(paths) if not q]
            raise RuntimeError(
                f"rank(s) {missing} could not set up the fd-exchange socket"
            )
        if sock is None:
            raise RuntimeError("fd-exchange socket was not initialized")

        fds[rank] = local_fd
        dst = paths[(rank + 1) % world_size]
        src = paths[(rank - 1) % world_size]
        in_flight = local_fd
        ring_error: Exception | None = None
        try:
            for step in range(1, world_size):
                _send_fd(sock, dst, in_flight)
                in_flight = _recv_fd(sock, src)
                fds[(rank - step) % world_size] = in_flight
        except Exception as e:
            ring_error = e
        oks: list[object] = [None] * world_size
        dist.all_gather_object(oks, ring_error is None, group=pg)
        if ring_error is not None:
            raise ring_error
        if not all(oks):
            failed = [i for i, ok in enumerate(oks) if not ok]
            raise RuntimeError(f"rank(s) {failed} failed the fd exchange")
        return fds
    except BaseException as e:
        for i, fd in enumerate(fds):
            if fd is not None and i != rank:
                os.close(fd)
        raise e
    finally:
        if sock is not None:
            sock.close()
        if sock_dir:
            _remove_dir(sock_dir)


def _send_fd(sock: socket.socket, dst_path: str, fd: int | None) -> None:
    ancdata = (
        []
        if fd is None
        else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
    )
    sock.sendmsg([_PAYLOAD], ancdata, 0, dst_path)


def _recv_fd(sock: socket.socket, src_path: str) -> int | None:
    msg, fds, flags, addr = socket.recv_fds(sock, len(_PAYLOAD), 1)
    error = ""
    if flags & socket.MSG_CTRUNC:
        error = (
            "SCM_RIGHTS control message truncated during the fd exchange; "
            "the peer's handle was lost in transit"
        )
    elif addr != src_path:
        error = (
            f"fd-exchange message came from {addr!r}, "
            f"expected the previous rank at {src_path!r}"
        )
    elif msg != _PAYLOAD or flags & socket.MSG_TRUNC:
        error = "fd-exchange message has an unexpected payload"
    if error:
        for fd in fds:
            os.close(fd)
        raise RuntimeError(error)
    return fds[0] if fds else None


def close_fds(fds: list[int | None]) -> None:
    for fd in fds:
        if fd is None:
            continue
        try:
            os.close(fd)
        except OSError:
            logger.debug("could not close ipc fd %d", fd)
