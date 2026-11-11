# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gc
import json
import os
import socket
import time
from contextlib import closing
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cuda.bindings import driver, runtime

import torchpyimpl.distributed.symmetric_memory as symm
from torchpyimpl.distributed.symmetric_memory import _stream_order


_N = 64
_OPS = ("barrier", "barrier_cas", "put_wait")
_OP_PAIRS = [(op, op) for op in _OPS] + [("barrier", "put_wait")]
_GATE_BLOCKED_PROBE_S = 1.0
_GATE_FINISH_TIMEOUT_S = 60.0


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init(rank: int, world_size: int, port: int, backend: str) -> torch.device:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device
    )
    symm.set_backend(backend)
    return device


def _spawn(worker: Any, backend: str, tmp_path: pathlib.Path) -> list[dict[str, Any]]:
    world_size = min(torch.cuda.device_count(), 8)
    out = str(tmp_path / "reports.json")
    mp.spawn(worker, args=(world_size, _free_port(), out, backend), nprocs=world_size)
    with open(out) as f:
        reports = json.load(f)["reports"]
    assert len(reports) == world_size
    return reports


def _report(r: dict[str, Any], out: str) -> None:
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, r)
    if dist.get_rank() == 0:
        with open(out, "w") as f:
            json.dump({"reports": gathered}, f)
    dist.barrier()


def _check(result: tuple[Any, ...]) -> Any:
    err, *rest = result
    if int(err) != 0:
        raise RuntimeError(f"CUDA call failed: {err}")
    return rest[0] if len(rest) == 1 else tuple(rest)


def _graph_edges(graph: torch.cuda.CUDAGraph) -> list[tuple[int, int]]:
    raw = graph.raw_cuda_graph()
    num_edges = _check(runtime.cudaGraphGetEdges(raw, numEdges=0))[-1]
    if num_edges == 0:
        return []
    srcs, dsts = _check(runtime.cudaGraphGetEdges(raw, numEdges=num_edges))[:2]
    return [(int(a), int(b)) for a, b in zip(srcs, dsts, strict=True)]


def _path_exists(edges: list[tuple[int, int]], src: int, dst: int) -> bool:
    successors: dict[int, list[int]] = {}
    for a, b in edges:
        successors.setdefault(a, []).append(b)
    seen: set[int] = set()
    stack = [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node not in seen:
            seen.add(node)
            stack.extend(successors.get(node, ()))
    return False


def _captured_order(
    stream: torch.cuda.Stream,
    launches_a: list[Callable[[], object]],
    launches_b: list[Callable[[], object]],
) -> tuple[bool, bool]:
    def tail(s: torch.cuda.Stream) -> int:
        info = _check(runtime.cudaStreamGetCaptureInfo(s.cuda_stream))
        deps, n = info[3], info[-1]
        assert n == 1, f"expected one node at the capture tail, got {n}"
        return int(deps[0])

    side = torch.cuda.Stream()
    marker = torch.zeros(1, device="cuda")
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        side.wait_stream(stream)
        for launch in launches_a:
            launch()
        a_last = tail(stream)
        marker.add_(1)
        marker_node = tail(stream)
        with torch.cuda.stream(side):
            launches_b[0]()
            b_first = tail(side)
            for launch in launches_b[1:]:
                launch()
        stream.wait_stream(side)
    edges = _graph_edges(graph)
    return _path_exists(edges, a_last, b_first), _path_exists(
        edges, marker_node, b_first
    )


def _gated_order(
    launches_a: list[Callable[[], object]],
    launches_b: list[Callable[[], object]],
    expect_ordered: bool,
    between: Callable[[], object] | None = None,
) -> tuple[bool, bool]:
    gates = torch.zeros(2, dtype=torch.int32, device="cuda")
    stream_a, stream_b, opener = (torch.cuda.Stream() for _ in range(3))
    for s in (stream_a, opener):
        s.wait_stream(torch.cuda.current_stream())
    geq = driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ

    def gate(i: int) -> None:
        ptr = gates[i].data_ptr()
        _check(driver.cuStreamWaitValue32(stream_a.cuda_stream, ptr, 1, geq))

    def open_gate(i: int) -> None:
        ptr = gates[i].data_ptr()
        _check(driver.cuStreamWriteValue32(opener.cuda_stream, ptr, 1, 0))

    def finished_within(event: torch.cuda.Event, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not event.query():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    b_done = torch.cuda.Event()
    try:
        with torch.cuda.stream(stream_a):
            gate(0)
            for launch in launches_a:
                launch()
            gate(1)
        if between is not None:
            between()
        with torch.cuda.stream(stream_b):
            for launch in launches_b:
                launch()
            b_done.record()
        probe = _GATE_BLOCKED_PROBE_S if expect_ordered else _GATE_FINISH_TIMEOUT_S
        while_gated = finished_within(b_done, probe)
        open_gate(0)
        after_open = finished_within(b_done, _GATE_FINISH_TIMEOUT_S)
    finally:
        open_gate(0)
        open_gate(1)
        torch.cuda.synchronize()
    return while_gated, after_open


def _launches(hdl: symm.SymmMemHandle, op: str) -> list[Callable[[], object]]:
    rank, world = hdl.rank, hdl.world_size
    nxt, prv = (rank + 1) % world, (rank - 1) % world

    def cas_barrier() -> None:
        mc = hdl.multicast_signal_pad_ptr
        hdl.multicast_signal_pad_ptr = 0
        try:
            hdl.barrier()
        finally:
            hdl.multicast_signal_pad_ptr = mc

    return {
        "barrier": [lambda: hdl.barrier()],
        "barrier_cas": [cas_barrier],
        "put_wait": [lambda: hdl.put_signal(nxt), lambda: hdl.wait_signal(prv)],
    }[op]


def _handle(group: dist.ProcessGroup) -> symm.SymmMemHandle:
    return symm.rendezvous(symm.empty(_N, device="cuda"), group)


def _structure_worker(
    rank: int, world_size: int, port: int, out: str, backend: str
) -> None:
    _init(rank, world_size, port, backend)
    world = dist.group.WORLD
    other = dist.new_group(list(range(world_size)))
    hdl_a, hdl_same, hdl_other = _handle(world), _handle(world), _handle(other)
    r: dict[str, Any] = {"rank": rank}

    main = torch.cuda.Stream()
    main.wait_stream(torch.cuda.current_stream())
    for op_a, op_b in _OP_PAIRS:
        for same in (True, False):
            a = _launches(hdl_a, op_a)
            b = _launches(hdl_same if same else hdl_other, op_b)
            with torch.cuda.stream(main):
                for launch in a + b:
                    launch()
            torch.cuda.synchronize()
            r[f"graph {op_a}->{op_b} same={same}"] = _captured_order(main, a, b)

    for same in (True, False):
        b_hdl = hdl_same if same else hdl_other
        r[f"eager same={same}"] = _gated_order(
            [lambda: hdl_a.barrier()],
            [lambda b_hdl=b_hdl: b_hdl.barrier()],
            expect_ordered=same,
        )

    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    def capture_barrier() -> None:
        with torch.cuda.stream(capture_stream):
            graph.capture_begin(capture_error_mode="thread_local")
            hdl_a.barrier()
            graph.capture_end()

    barrier = [lambda: hdl_a.barrier()]
    r["eager across capture"] = _gated_order(
        barrier, barrier, expect_ordered=True, between=capture_barrier
    )

    first = torch.cuda.CUDAGraph()
    with torch.cuda.graph(first, stream=capture_stream):
        hdl_a.barrier()
    with torch.cuda.stream(main):
        hdl_a.barrier()
    torch.cuda.synchronize()
    r["graph second capture"] = _captured_order(main, barrier, barrier)
    r["eager after captures"] = _gated_order(barrier, barrier, expect_ordered=True)

    transient = dist.new_group(list(range(world_size)))
    _handle(transient).barrier()
    torch.cuda.synchronize()
    r["state while group lives"] = transient in _stream_order._states
    before = len(_stream_order._states)
    dist.destroy_process_group(transient)
    del transient
    gc.collect()
    r["states dropped with group"] = before - len(_stream_order._states)
    _report(r, out)
    dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.peer_access
@pytest.mark.parametrize("backend", ["CUDA", "NCCL"])
def test_ordering_structure(backend: str, tmp_path: pathlib.Path) -> None:
    if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1":
        pytest.skip("stream gates deadlock with one hardware queue")
    for r in _spawn(_structure_worker, backend, tmp_path):
        rank = r["rank"]
        for op_a, op_b in _OP_PAIRS:
            for same in (True, False):
                key = f"graph {op_a}->{op_b} same={same}"
                after_a, after_marker = r[key]
                assert after_a == same, f"rank {rank}: {key}: ordered={after_a}"
                assert not after_marker, f"rank {rank}: {key}: waits behind a"
        for same in (True, False):
            while_gated, after_open = r[f"eager same={same}"]
            assert while_gated != same, f"rank {rank}: eager same={same}"
            assert after_open, f"rank {rank}: eager same={same}: waits behind a"
        assert r["eager across capture"] == [False, True], rank
        assert r["graph second capture"] == [True, False], rank
        assert r["eager after captures"] == [False, True], rank
        assert r["state while group lives"], rank
        assert r["states dropped with group"] == 1, rank
