# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""POSIX fd all-gather for CUDA-IPC ``POSIX_FILE_DESCRIPTOR`` handles.

Each rank binds an ``AF_UNIX`` datagram socket; fds then travel around a ring
with ``SCM_RIGHTS`` in ``world_size - 1`` steps. Single node only.
"""

from __future__ import annotations

import array
import logging
import os
import socket
import tempfile
import uuid

import torch.distributed as dist


logger: logging.Logger = logging.getLogger(__name__)

# Long enough to cover a slow rank arriving at the exchange, short enough that a
# genuinely stuck peer fails the job instead of hanging it forever.
_RECV_TIMEOUT_S: float = 300.0
_PAYLOAD: bytes = b"fd"


def _socket_path() -> str:
    """A unique path for this rank's socket. ``AF_UNIX`` paths are capped at
    ~108 bytes, so keep it short: the temp dir plus a pid + random suffix."""
    return os.path.join(
        tempfile.gettempdir(), f"torchpyimpl-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )


def all_gather_fds(
    local_fd: int | None,
    pg: dist.ProcessGroup | None,
) -> list[int | None]:
    """Collective over ``pg``. Return every rank's fd, indexed by group rank.
    ``pg=None`` means a single rank. If one rank fails, every rank raises.

    ``local_fd`` may be ``None`` for a rank with nothing to contribute (used to
    broadcast one rank's multicast handle: only the creator passes an fd, and
    callers read index ``root``). ``None`` relays through the ring like any other
    slot.

    The returned fds are **owned by the caller** and must all be closed once
    imported (see :func:`close_fds`); ``local_fd`` is not closed here and is
    returned at index ``rank``.
    """
    if pg is None:
        return [local_fd]
    rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return [local_fd]

    # Set up before the collective below, so every peer is already listening by
    # the time anyone learns its address — no connect/send retry needed. This is
    # also a per-rank failure point (EMFILE, an unwritable temp dir, an AF_UNIX
    # path over the ~108-byte limit) sitting in front of this function's own
    # collective, so a rank that fails it still joins that collective with an
    # empty path rather than walking out and leaving its peers inside.
    path = _socket_path()
    sock: socket.socket | None = None
    setup_error: Exception | None = None
    bound = False
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(path)
        bound = True
        sock.settimeout(_RECV_TIMEOUT_S)
    except Exception as e:
        setup_error = e

    fds: list[int | None] = [None for _ in range(world_size)]
    try:
        paths: list[str] = [""] * world_size
        dist.all_gather_object(paths, path if bound else "", group=pg)
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
        in_flight = local_fd
        ring_error: Exception | None = None
        try:
            for step in range(1, world_size):
                _send_fd(sock, dst, in_flight)
                in_flight = _recv_fd(sock)
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
        # BaseException, not Exception: the callers guard their own export the
        # same way, so a KeyboardInterrupt or cancellation mid-ring must not
        # leak the fds we have already received either.
        for i, fd in enumerate(fds):
            if fd is not None and i != rank:
                os.close(fd)
        raise e
    finally:
        if sock is not None:
            sock.close()
        # Only unlink what this call created: a failed bind means the path is
        # somebody else's. Best-effort otherwise — the path is gone once every
        # peer has sent to us, but a crashed peer could leave it behind.
        if bound:
            try:
                os.unlink(path)
            except OSError:
                logger.debug("could not unlink ipc socket %s", path)


def _send_fd(sock: socket.socket, dst_path: str, fd: int | None) -> None:
    """Send ``fd`` to ``dst_path``. A ``None`` sends the payload with no control
    message, which the receiver reads back as ``None``.

    Uses ``sendmsg`` directly rather than ``socket.send_fds``: the stdlib helper
    accepts an ``address`` argument but drops it on the floor, which on an
    unconnected datagram socket fails with ``ENOTCONN``. Connecting the socket
    instead is not an option — it would also filter out the datagrams we need to
    receive from the *other* neighbour in the ring."""
    ancdata = (
        []
        if fd is None
        else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
    )
    sock.sendmsg([_PAYLOAD], ancdata, 0, dst_path)


def _recv_fd(sock: socket.socket) -> int | None:
    """Receive one fd, or ``None`` if the sender deliberately passed nothing.

    ``MSG_CTRUNC`` means the kernel dropped the control message — the fd is gone
    and the slot would otherwise be indistinguishable from a deliberate
    ``None``, surfacing much later as an opaque driver error on import."""
    _msg, fds, flags, _addr = socket.recv_fds(sock, len(_PAYLOAD), 1)
    if flags & socket.MSG_CTRUNC:
        # Truncation drops *some* control data, so a descriptor may still have
        # arrived alongside the flag. Close it before raising.
        for fd in fds:
            os.close(fd)
        raise RuntimeError(
            "SCM_RIGHTS control message truncated during the fd exchange; "
            "the peer's handle was lost in transit"
        )
    return fds[0] if fds else None


def close_fds(fds: list[int | None]) -> None:
    """Close every fd in ``fds``, including this rank's own export.

    Both sides of the exchange are transport artifacts: the driver dups an fd
    on import, and the exporter's block stays alive through the
    ``CUmemGenericAllocationHandle`` from ``cuMemCreate``, not through the fd.
    Leaking these would exhaust the fd table one rendezvous at a time."""
    for fd in fds:
        if fd is None:
            continue
        try:
            os.close(fd)
        except OSError:
            logger.debug("could not close ipc fd %d", fd)
