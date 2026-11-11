# Public API

`torchpyimpl/distributed/symmetric_memory/_api.py`, re-exported from `torchpyimpl.distributed.symmetric_memory`.
Layers underneath: [`allocator.md`](allocator.md). Synchronization primitives:
[`signal.md`](signal.md).

## Torch parity

Function names, argument order and offset conventions follow
`torch.distributed._symmetric_memory`. Accessors are module functions that take
the handle as their first argument (torch exposes them as methods on the
handle). `barrier`, `put_signal` and `wait_signal` are methods on the handle, as
in torch. The package does not export them as module functions: torch's
module-level `put_signal` and `wait_signal` have other signatures and semantics.
The handle carries the same pointer fields (`buffer_ptrs`, `signal_pad_ptrs`,
`multicast_ptr`, ...), so kernels that read those fields work unchanged. Drift
in an offset convention or a pad view would produce wrong numbers rather than an
error.

## Two entry paths, one allocator

```
  MemPool path                           standalone path
  ---------------------------            ---------------
  empty(...)                             _symm_mem.alloc(nbytes, device)
        |  allocates through the                |  raw allocator, no MemPool
        |  symmetric MemPool                    |
        v                                       v
  rendezvous(tensor, group)              _symm_mem.rendezvous(ptr, ..., comm)
        |                                       |
        +---------------+-----------------------+
                        v
                  SymmMemHandle
```

The first is the recommended path. `empty()` returns an ordinary tensor whose
storage came from the symmetric pool: anything taking a tensor takes this one,
and the symmetry is a property of the memory, not of a tensor subclass.
`rendezvous()` then turns it into a handle.

The standalone path is for callers managing raw allocations without a `MemPool`.
Both converge on the same allocator and the same handle.

## Exports

| function | purpose |
|---|---|
| `set_backend(name)` | `"CUDA"` (default) or `"NCCL"`; before the first allocation |
| `set_signal_pad_size(n)` / `get_signal_pad_size()` | signal-pad bytes; frozen at the first allocation |
| `empty(*size, dtype, device)` | tensor from the symmetric `MemPool` |
| `rendezvous(tensor, group)` | collective; `group` is a process group or its name; returns a cached `SymmMemHandle` |
| `free_pool(device, group=None)` | tear down peer state and release the device's pool; with `group`, the ranks agree first; returns a `FreePoolStatus` |
| `get_buffer`, `get_remote_tensor`, `get_signal_pad` | views over peer memory |
| `has_multicast_support(handle)` | whether the allocation has a multimem pointer |

`SymmMemHandle` methods, for signal-pad synchronization:

| method | purpose |
|---|---|
| `barrier(channel=0, timeout_ms=None)` | barrier across the handle's ranks |
| `put_signal(dst_rank, channel=0, timeout_ms=None)` | raise this rank's slot in `dst_rank`'s pad |
| `wait_signal(src_rank, channel=0, timeout_ms=None)` | wait for `src_rank`'s signal, then reset the slot |

The handle also carries `group_name`, the group it was rendezvoused on, as
torch's `hdl.group_name`. A buffer rendezvoused on two groups has two handles,
one per group. The handle holds its group weakly.

## CUDA graphs

Allocation and the first rendezvous of an allocation on a group cannot be
captured, and raise while the current stream is capturing a graph, as in torch.
Allocation maps the block, zeroes the signal pad and synchronizes the device.
`empty()` reaches it only on a pool miss. The first rendezvous exchanges
handles across ranks. Allocate and rendezvous before starting capture.

Later rendezvous calls on the same allocation and group reuse the cached peer
state and are capture-safe. That includes new views of the allocation. So are
`barrier`, `put_signal` and `wait_signal`. Every rank must replay a captured
graph the same number of times, because each replay performs that rank's side
of the synchronization.

## The handle contract

| accessor | returns | offset convention |
|---|---|---|
| `get_buffer(h, peer, sizes, dtype, storage_offset)` | peer's data buffer | from the peer's buffer **base**, plus `storage_offset` |
| `get_remote_tensor(h, peer, sizes, dtype)` | peer's data buffer | from the peer's base **plus this view's own offset** |
| `get_signal_pad(h, peer, sizes, dtype, storage_offset)` | peer's signal pad | whole pad by default |

The first two differ exactly as `_SymmetricMemory::get_buffer` and
`get_remote_tensor` do, and the difference shows up as soon as a caller
rendezvouses a *slice*: one addresses the allocation, the other the view.
Requests are bounds-checked against the peer allocation, and views are labelled
with the allocation's device.

All three return **non-owning** tensors over peer memory. The peer's allocation
owns the storage; using a view after that allocation is freed is a use-after-free
no refcount will catch.

## Backend and pad selection

`set_backend` picks how memory is shared.
Backend selection must precede the first allocation — the choice is baked into
every live allocation's physical memory.

`set_signal_pad_size()` / `get_signal_pad_size()` set and read the pad size,
which freezes at the first allocation.
