# NCCL backend (`"NCCL"`)

| Layer | Files |
|---|---|
| borrowing the communicator | `torchpyimpl/distributed/symmetric_memory/_comm.py` |
| NCCL symmetric-memory backend | `torchpyimpl/distributed/symmetric_memory/backends/_nccl.py` |

The allocator machinery this plugs into is [`allocator.md`](allocator.md); the
public surface is [`api.md`](api.md).

## 1. What it does

`NCCLAllocator` gets symmetric memory from NCCL itself. Physical memory comes
from `ncclMemAlloc`, the block is registered as a **symmetric window** on a
communicator, and the per-rank pointers are read back out of that window handle.

There is no handle exchange to arrange: registration *is* the collective, and
NCCL resolves peer addressability. The backend's job is to register at the right
granularity, in an order every rank agrees on, and to hand back the pointer
arrays.

The block is laid out signal-pad-first, so a window's peer base is that peer's
signal-pad base and its data buffer is `+ buffer_offset()`. See
[`allocator.md`](allocator.md) §6.

This is the only module that imports `nccl4py`; the allocator base stays
backend-agnostic and imports it lazily.

## 2. Borrowing the communicator

The backend never creates an `ncclComm_t` — it borrows the one PyTorch already
has, so symmetric memory rides the same communicator as the collectives around
it.

`get_group_info(pg, device)` resolves everything a rendezvous needs to know
about a group — the `ncclComm_t` from the process group backend's `comm_ptr`,
plus the rank, the world size and the group's collective timeout — and memoizes
the whole `GroupInfo` on `(id(pg), device index, require_comm)`, validated
through a weakref to the group. `get_nccl_comm_ptr` is a thin accessor for the pointer alone.

The default `"nccl"` process group backend creates its communicator in its
constructor, so `comm_ptr` is valid as soon as the group exists and does not
depend on the current device. A null pointer raises.

## 3. Allocating physical memory

`ncclMemAlloc`. Two properties of it shape the backend:

**It is VMM-backed.** The driver therefore knows the extent of an allocation, so
`allocation_base` is just `cuMemGetAddressRange` — recovering the base of an
interior pointer costs one call and no side table.

**It allocates on the *current* device**, while the record is filed under the
requested `device`. `alloc_physical` wraps the call in `torch.cuda.device(device)`
for that reason: without the guard, an allocation requested for a non-current
device is recorded against the wrong one, which breaks both the per-device
teardown grouping and the peer mapping.

## 4. Rendezvous

Registration is the collective, so rendezvous is short:

```
  register the block as a symmetric window   (collective — every rank,
        |                                     matching blocks, same order)
  multimem pointers at the pad and the data buffer
        |
  for each peer: get_peer_device_pointer(rank)
        |
  vote over the process group                (ok, multicast)
        v
  _PeerAllocInfo
```

**Registering.** `RegisteredWindowHandle(Comm(comm_ptr), alloc_base, block_size,
flags=CollSymmetric)` is `ncclCommWindowRegister` underneath. It covers the whole
block — pad and data buffer together — and it is collective: every rank must
register matching allocations in the same order. That ordering requirement is
what drives the deferred-release design in [`allocator.md`](allocator.md) §3.

**Peer pointers.** `window.get_peer_device_pointer(rank)`
(`ncclGetPeerDevicePointer`) returns each rank's base for this window. A null
means the peer is not reachable over LSA, which is raised rather than stored — a
zero pointer would otherwise surface much later as a bad access inside a kernel.

**Multimem.** `window.get_lsa_multimem_device_pointer(off)` gives the LSA multimem
address at the data buffer (`multicast_ptr`) and, at offset 0, at the signal pad
(`multicast_signal_pad_ptr`). Builds and hardware without one return nothing, in
which case both are 0 and callers use the peer pointers. The first query on a
communicator can be collective (it creates the LSA multimem team), so every rank
makes it before any step that can fail on one rank alone; an error means no
multicast.

**Setup failure.** After the local steps, the ranks vote over the process group.
If one rank failed, every rank closes its window and raises. Deregistration is
local to the rank, so all ranks keep the same windows. Multicast is used only if
every rank has it.

The window handle and a weak reference to its process group are stored in
`_PeerAllocInfo.window`, which ties the window's lifetime to the allocation
record.

## 5. Teardown

`close_peers` closes the window with `ncclCommWindowDeregister`. Deregistration
is local to the rank, but every rank must close the same windows: NCCL gives
each new window an offset that depends on the windows the rank has, and binds
multicast at that offset. After a CUDA error
(`should_skip_cuda_cleanup`) it returns `DEFERRED` and is retried later; if the
close raises, it is logged and returns `FAILED`, and the record is retained as
leaked rather than raising out of teardown. A retry stays `FAILED`: nccl4py marks
the window closed before it deregisters.

If the process group was destroyed, NCCL freed the window with the communicator.
`close_peers` then does not deregister it, and `rendezvous` does not reuse it
(`peers_alive`), because a new communicator can have the same address. An aborted
group that was not destroyed still counts as alive.

`free_pool(device, group)` agrees across the group before the teardown, so either
every rank closes its windows or none does.

`free_physical` is `ncclMemFree`, under the owning device for the same reason
`alloc_physical` pins the device, with the same deferred and failed handling.

## 6. Hooks

What this backend supplies to `SymmetricMemoryAllocatorBase`
([`allocator.md`](allocator.md) §7):

| Hook | Implementation |
|---|---|
| `alloc_physical` | `ncclMemAlloc`, pinned to the record's device |
| `free_physical` | `ncclMemFree` |
| `allocation_base` | `cuMemGetAddressRange` |
| `establish_peers` | register a symmetric window, then read peer + multimem pointers from it |
| `close_peers` | close the window (`ncclCommWindowDeregister`) |
| `peers_alive` | false once the window's process group is destroyed |

## 7. Invariants

1. Window registration is collective: every rank must rendezvous matching
   allocations in the same order. Deregistration is local, but every rank must
   deregister the same windows before the next registration.
2. The communicator is borrowed from the process group, never created.
3. `ncclMemAlloc` allocates on the current device; `alloc_physical` pins it to the
   device the record is filed under.
4. A null peer pointer means no LSA reachability — an error, not a fallback.
