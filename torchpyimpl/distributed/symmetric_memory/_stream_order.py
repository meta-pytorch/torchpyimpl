# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import threading
import weakref
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from cuda.bindings import driver


if TYPE_CHECKING:
    from types import TracebackType

    from torchpyimpl.distributed.symmetric_memory._symm_mem import SymmMemHandle


logger: logging.Logger = logging.getLogger(__name__)


class _LastOp:
    __slots__ = ("done", "raw", "stream")

    def __init__(self) -> None:
        self.done = torch.cuda.Event()
        self.stream: torch.cuda.Stream | None = None
        self.raw: int | None = None

    def reset(self) -> None:
        self.stream = self.raw = None


class _State:
    __slots__ = ("capture", "capture_id", "eager", "lock")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.eager = _LastOp()
        self.capture_id: int | None = None
        self.capture = _LastOp()


_states: weakref.WeakKeyDictionary[dist.ProcessGroup, dict[int, _State]] = (
    weakref.WeakKeyDictionary()
)
_states_lock = threading.Lock()


def _state(group: dist.ProcessGroup, device: int) -> _State:
    with _states_lock:
        per_device = _states.get(group)
        if per_device is None:
            per_device = _states[group] = {}
        state = per_device.get(device)
        if state is None:
            state = per_device[device] = _State()
        return state


def _capture_id(raw_stream: int) -> int:
    err, _status, capture_id, *_ = driver.cuStreamGetCaptureInfo(
        driver.CUstream(raw_stream)
    )
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuStreamGetCaptureInfo failed: {err}")
    return int(capture_id)


class group_stream_guard:
    __slots__ = ("_device", "_last", "_state")

    def __init__(self, handle: SymmMemHandle) -> None:
        self._state: _State | None = None
        self._last: _LastOp | None = None
        self._device = torch.cuda.current_device()
        group = None if handle.group_ref is None else handle.group_ref()
        if group is not None:
            self._state = _state(group, self._device)

    def __enter__(self) -> None:
        state = self._state
        if state is None:
            return
        raw = torch._C._cuda_getCurrentRawStream(self._device)
        state.lock.acquire()
        try:
            last = state.eager
            if torch.cuda.is_current_stream_capturing():
                capture_id = _capture_id(raw)
                if state.capture_id != capture_id:
                    state.capture_id = capture_id
                    state.capture.reset()
                last = state.capture
            if last.raw != raw:
                cur = torch.cuda.current_stream(self._device)
                if last.stream is not None:
                    cur.wait_event(last.done)
                last.stream, last.raw = cur, raw
        except BaseException:
            state.lock.release()
            raise
        self._last = last

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        state, last = self._state, self._last
        if state is None or last is None:
            return
        try:
            last.done.record(last.stream)
        except Exception:
            logger.warning("symm_mem: failed to record the stream-ordering event")
            last.reset()
        finally:
            state.lock.release()
