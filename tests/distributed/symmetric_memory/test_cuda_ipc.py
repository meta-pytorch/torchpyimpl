# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the CUDA-IPC backend and the backend registry."""

from __future__ import annotations

import os
from contextlib import nullcontext

import pytest
import torch
from cuda.bindings import driver

import torchpyimpl.distributed.symmetric_memory._symm_mem as sm
from torchpyimpl.distributed.symmetric_memory.backends import _cuda_ipc
from torchpyimpl.distributed.symmetric_memory.backends._cuda_ipc import IpcHandleType


def test_set_active_backend_validation() -> None:
    assert sm.get_active_backend() == "CUDA"
    sm.set_active_backend("CUDA")
    with pytest.raises(ValueError):
        sm.set_active_backend("NVSHMEM")
    assert sm.get_active_backend() == "CUDA"


def test_cuda_ipc_transport_env_is_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_ipc.reset_transport()
    monkeypatch.setitem(os.environ, "TORCHPYIMPL_CUDA_IPC_TRANSPORT", "bogus")
    try:
        with pytest.raises(ValueError):
            _cuda_ipc.get_transport(0)
    finally:
        _cuda_ipc.reset_transport()


def _peer(host: str, transport: IpcHandleType, device: int = 0) -> dict[str, object]:
    return {
        "size": 4096,
        "device": device,
        "host": host,
        "multicast": True,
        "transport": transport.value,
    }


def test_posix_fd_transport_rejects_a_multi_host_group() -> None:
    validate = _cuda_ipc.CUDAIPCAllocator._validate
    with pytest.raises(RuntimeError, match="single-node only"):
        validate(
            [
                _peer("host-a", IpcHandleType.POSIX_FD),
                _peer("host-b", IpcHandleType.POSIX_FD),
            ],
            0,
            4096,
            IpcHandleType.POSIX_FD,
        )
    assert (
        validate(
            [
                _peer("host-a", IpcHandleType.FABRIC),
                _peer("host-b", IpcHandleType.FABRIC),
            ],
            0,
            4096,
            IpcHandleType.FABRIC,
        )
        is None
    )


def test_validate_rejects_ranks_on_different_transports() -> None:
    validate = _cuda_ipc.CUDAIPCAllocator._validate
    with pytest.raises(RuntimeError, match="disagree on the CUDA-IPC transport"):
        validate(
            [
                _peer("host-a", IpcHandleType.FABRIC, device=0),
                _peer("host-a", IpcHandleType.POSIX_FD, device=1),
            ],
            0,
            4096,
            IpcHandleType.FABRIC,
        )


def test_validate_rejects_two_ranks_on_one_gpu() -> None:
    validate = _cuda_ipc.CUDAIPCAllocator._validate
    with pytest.raises(RuntimeError, match="same GPU"):
        validate(
            [
                _peer("host-a", IpcHandleType.FABRIC, device=0),
                _peer("host-a", IpcHandleType.FABRIC, device=0),
            ],
            0,
            4096,
            IpcHandleType.FABRIC,
        )


def test_vmm_cleanup_continues_after_unmap_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_unmap(_base: int, _size: int) -> tuple[object]:
        calls.append("unmap")
        raise RuntimeError("injected unmap failure")

    def free_address(_base: int, _size: int) -> tuple[object]:
        calls.append("address_free")
        return (driver.CUresult.CUDA_SUCCESS,)

    def release_handle(_handle: object) -> tuple[object]:
        calls.append("release")
        return (driver.CUresult.CUDA_SUCCESS,)

    monkeypatch.setattr(_cuda_ipc.driver, "cuMemUnmap", fail_unmap)
    monkeypatch.setattr(_cuda_ipc.driver, "cuMemAddressFree", free_address)
    monkeypatch.setattr(_cuda_ipc.driver, "cuMemRelease", release_handle)

    assert _cuda_ipc.cu_vmm_free(0x1000, 4096, object()) is False
    assert calls == ["unmap", "address_free", "release"]


def test_failed_peer_cleanup_retains_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _cuda_ipc._CudaMapping(0x1000, 4096, object(), 0)
    state = _cuda_ipc._CudaPeerState(imported=[mapping], multicast=None)
    empty = torch.empty(0, dtype=torch.int64)
    pai = sm._PeerAllocInfo(state, [], [], empty, empty, 0, device=0)
    monkeypatch.setattr(
        _cuda_ipc._cuda, "should_skip_cuda_cleanup", lambda _device: False
    )
    monkeypatch.setattr(_cuda_ipc.torch.cuda, "device", lambda _idx: nullcontext())
    monkeypatch.setattr(_cuda_ipc, "cu_vmm_free", lambda *_args: False)

    assert _cuda_ipc.CUDAIPCAllocator().close_peers(pai) is sm.CleanupStatus.FAILED
    assert pai.window is state


def test_rendezvous_oom_reuses_existing_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mine = _cuda_ipc._CudaMapping(0x100000, 4096, object(), 0)
    imported = [_cuda_ipc._CudaMapping(0x200000, 4096, object(), 0)]
    agreements: list[bool] = []
    freed: list[int] = []

    monkeypatch.setattr(_cuda_ipc.dist, "get_rank", lambda _pg: 0)
    monkeypatch.setattr(
        _cuda_ipc, "get_transport", lambda _dev: _cuda_ipc.IpcHandleType.FABRIC
    )
    monkeypatch.setattr(_cuda_ipc, "cu_mem_export", lambda *_args: b"handle")
    monkeypatch.setattr(_cuda_ipc, "cu_device_supports_multicast", lambda _dev: False)

    def gather(out: list[object], payload: object, group: object) -> None:
        out[0] = payload

    monkeypatch.setattr(_cuda_ipc.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        _cuda_ipc.CUDAIPCAllocator,
        "_agree_ready_to_exchange",
        lambda *_args: b"handle",
    )
    monkeypatch.setattr(
        _cuda_ipc.CUDAIPCAllocator,
        "_import_peers",
        lambda *_args: ([mine.base], imported),
    )
    monkeypatch.setattr(
        _cuda_ipc.CUDAIPCAllocator,
        "_init_multicast",
        lambda *_args: pytest.fail("multicast setup ran after pointer-tensor OOM"),
    )

    def agree(_pg: object, _world_size: int, ok: bool) -> bool:
        agreements.append(ok)
        return ok

    def free_mapping(base: int, _size: int, _handle: object) -> bool:
        freed.append(base)
        return True

    monkeypatch.setattr(_cuda_ipc, "_all_agree", agree)
    monkeypatch.setattr(_cuda_ipc, "cu_vmm_free", free_mapping)

    def fail_tensor(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise torch.OutOfMemoryError("injected")

    monkeypatch.setattr(_cuda_ipc.torch, "tensor", fail_tensor)
    rec = sm.AllocRecord(
        alloc_base=mine.base,
        block_size=mine.size,
        buffer_size=1024,
        device=0,
        handle=mine,
    )

    with pytest.raises(torch.OutOfMemoryError, match="injected"):
        _cuda_ipc.CUDAIPCAllocator().establish_peers(
            rec,
            comm_ptr=0,
            world_size=1,
            device=torch.device("cuda:0"),
            group=object(),
        )

    assert agreements == [False]
    assert freed == [imported[0].base]
