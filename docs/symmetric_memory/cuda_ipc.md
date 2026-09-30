# CUDA-IPC backend (`"CUDA"`)

| Layer | Files |
|---|---|
| CUDA VMM + IPC | `torchpyimpl/distributed/symmetric_memory/backends/_cuda_ipc.py` |
| multicast / LSA multimem | `torchpyimpl/distributed/symmetric_memory/backends/_cuda_ipc.py`, `torchpyimpl/distributed/symmetric_memory/_multimem.py` |
| POSIX-fd transport | `torchpyimpl/distributed/symmetric_memory/backends/_cuda_ipc.py`, `torchpyimpl/distributed/symmetric_memory/backends/_ipc_fd.py` |

The allocator machinery this plugs into is [`allocator.md`](allocator.md); the
public surface is [`api.md`](api.md).

## 1. What it does

`CUDAIPCAllocator` builds symmetric memory directly on the CUDA driver. Every
rank allocates its own block through the Virtual Memory Management API with an
exportable handle, the ranks trade those handles, and each rank maps every peer's
block into its own address space. After that the peer buffers are ordinary device
pointers: reads and writes go over NVLink or PCIe peer-to-peer with no further
involvement from the driver or the process group.

How the handles travel depends on their kind — a fabric handle is 64 bytes and
pickles, a POSIX file descriptor does not and has to be passed over a Unix socket
(§3).

Where the hardware supports it, the same allocation also gets a **multicast**
object — one address whose stores land in every rank's copy and whose loads
reduce across them.

The block a rank allocates is laid out signal-pad-first, so a single physical
allocation carries both the synchronization pad and the data buffer:

```
  alloc_base                          buffer_ptr = alloc_base + buffer_offset
  |                                   |
  v                                   v
  [ signal pad | barrier state ]      [ data buffer ]
  <---------- buffer_offset --------->
```

`buffer_offset()` is a process-global constant, which is what lets an interior
pointer be resolved back to its allocation in O(1) — ask the driver for the
allocation base, add the constant. See [`allocator.md`](allocator.md) §6.

## 2. Allocating physical memory

CUDA's VMM API separates the physical pages from the address they appear at, in
three steps:

```
  cuMemCreate            allocate physical pages, tagged exportable
  cuMemAddressReserve    reserve a virtual address range
  cuMemMap + SetAccess   bind the two, grant this device read/write
```

`alloc_physical` runs all three and hands the allocator base a `_CudaMapping`
holding `(base, size, handle, device)` — the VA it mapped, the rounded size, the
`CUmemGenericAllocationHandle` that owns the pages, and the owning device.

Two details here constrain everything downstream:

**The handle type is fixed at creation.** `requestedHandleTypes` is part of the
`cuMemCreate` property block, so a block's shareable-handle kind cannot change
while it is live, and ranks that chose differently cannot import each other's
memory. This is why the transport is resolved once and frozen (§3).

**The size is rounded up twice.** First to the driver's recommended allocation
granularity, then — on hardware with multicast support — to the multicast
granularity as well, because `cuMulticastBindMem` requires a size that is a
multiple of it. Rounding at allocation time means multicast can be bound later
without reallocating. That rounded size, not the caller's request, is what peers
must map.

`allocation_base` is `cuMemGetAddressRange`: the driver already knows the extent
of a VMM allocation, so recovering the base of an interior pointer costs one
call and no bookkeeping.

## 3. Transports

A shareable handle comes in two kinds, `IpcHandleType.FABRIC` and
`IpcHandleType.POSIX_FD`, and they differ in what it takes to move one between
processes:

| | `fabric` | `fd` |
|---|---|---|
| handle | 64 opaque bytes | a POSIX file descriptor |
| exchange | rides `dist.all_gather_object` | `SCM_RIGHTS` over `AF_UNIX` |
| hardware | fabric-capable (GB200/GB300, IMEX-enabled H100 NVSwitch) | any P2P-capable GPU |
| reach | across nodes in the fabric | **single node only** |

`get_transport(device)` picks one on first use — fabric when the device supports
it, otherwise fds — and **freezes it for the process**, because §2's handle type
is baked into every block at creation. `TORCHPYIMPL_CUDA_IPC_TRANSPORT=fabric|fd`
forces the choice, and an unsupported forced value is rejected up front rather
than failing later in the driver.

### Passing a file descriptor

An fd is a process-local kernel object: it is meaningless in another process and
cannot be pickled. The only way to hand one over is `sendmsg` with an
`SCM_RIGHTS` control message over a Unix domain socket, which makes the kernel
install a *new* fd for the same open file in the receiver.
`torchpyimpl/distributed/symmetric_memory/backends/_ipc_fd.py` builds an all-gather out of that:

```
  every rank binds an AF_UNIX/SOCK_DGRAM socket at a unique path
  all_gather_object the paths          (bind first, so every peer is listening)
  ring, world_size - 1 steps:
      send what you currently hold to rank + 1
      receive rank - k's fd from rank - 1
  all_gather_object an ok flag          (if one rank failed, every rank raises)
```

Two properties make the ring safe. It is **datagram**-based, so every rank sends
before it receives and the kernel buffers the tiny message — no deadlock, no
connect/retry. And it is `O(world_size)` steps with one datagram each, rather
than a quadratic mesh of sockets.

Binding is itself a per-rank failure point (`EMFILE`, an unwritable temp dir, a
path past the ~108-byte `AF_UNIX` limit) sitting in front of the path gather. A
rank that fails it therefore still *joins* that gather, contributing an empty
path, and every rank raises together — the same reasoning as the votes in §4,
applied inside the ring because the ring owns the collective.

**The received fds belong to the caller.** The driver dups an fd on import, and
the exporter's block stays alive through its `CUmemGenericAllocationHandle`, not
through the fd — so both sides of the exchange are transport litter that has to
be closed, or a rendezvous loop exhausts the fd table one round at a time. Every
path that obtains fds closes them in a `finally`.

Two failures are made explicit rather than left to surface later: a truncated
control message (`MSG_CTRUNC`) means the kernel dropped an fd, which is otherwise
indistinguishable from a rank deliberately passing nothing; and a missing slot in
an exchange where every rank contributes is rejected before it can reach
`cuMemImportFromShareableHandle` as an opaque driver error.

Because fds do not cross hosts, `_validate` rejects a multi-host group on this
transport at rendezvous (§4).

## 4. Rendezvous

Rendezvous turns one rank's local allocation into a set of pointers covering
every rank:

```
  local                                    collective
  -----                                    ----------
  payload = {handle?, size, device,
             host, multicast, transport}
        |
        +------------------------------->  all_gather_object
        |                              <-- every rank's payload
  _validate
        |
  owning-device check
  export (fd transport only)
        +------------------------------->  agree: ready to exchange
        |
  handle exchange                          (fabric: already gathered
        |                                   fd: the SCM_RIGHTS ring)
  for each peer: import + map RW           (local)
        +------------------------------->  agree: everyone imported
        |
  buffers[] / signal_pads[] arrays
        |
  multicast setup                          (§5)
        v
  _PeerAllocInfo
```

**The payload.** Each field is there to make a specific failure impossible:

- `handle` — the exported shareable handle on fabric, where it pickles. On the fd
  transport it is `None` and the real exchange happens in the ring.
- `size` — the granularity-rounded `mapped_size`. It travels so importers map
  exactly what the exporter created; recomputing the rounding independently
  would be a silent size mismatch.
- `host` + `device` — together they identify a physical GPU. Two ranks reporting
  the same pair means a misconfigured `LOCAL_RANK` or `set_device`, which would
  otherwise corrupt memory quietly rather than fail. On the fd transport, two
  *different* hosts is equally fatal, and rejected here too.
- `multicast` — whether this rank's device supports it, so the multimem path can
  be gated on unanimity.
- `transport` — the kind this rank resolved (§3). It is resolved independently
  per process, so nothing else would catch a disagreement: fd ranks would sit in
  the `SCM_RIGHTS` ring while fabric ranks skipped it, and the two mismatched
  collectives would pair up. `_validate` rejects a mixed group outright.

**The owning-device check.** Peer blocks are mapped with RW access granted to
the rendezvous device, so rendezvousing on a device other than the one that owns
the allocation would map it for the wrong GPU. `_validate` structurally cannot
catch this — it compares the gathered devices, which all agree — so it is checked
directly against `mine.device`.

**Agreeing before exchanging.** The two votes in the diagram are what keep a
one-rank failure from becoming a hang. Everything between the payload gather and
the exchange that can fail on a single rank — the owning-device check, and the fd
export — happens together in `_agree_ready_to_exchange`, which votes before
anyone enters the exchange. Without that vote a rank which failed to export would
walk out while its peers sat in the ring's own `all_gather_object`; the two
mismatched object collectives pair up, and the ranks diverge silently instead of
erroring. The second vote does the same job for the local import loop, which runs
after the exchange but before the multicast collectives.

**Importing peers.** For each peer, `cuMemImportFromShareableHandle` turns its
handle into a local handle, then the same reserve-and-map sequence as §2 gives it
an address. That address is chosen by *this* rank's driver, so the same physical
buffer sits at a different virtual address on every rank. Nothing assumes
otherwise: the result is the per-peer pointer arrays (`buffers[]`,
`signal_pads[]`, indexed by rank), and every accessor indexes through them.

Since the block is signal-pad-first, a mapping's base *is* its signal-pad base,
and the data buffer is that plus `buffer_offset()`.

**Failure rolls back.** Until a `_PeerAllocInfo` is returned the caller owns
nothing, so a failure partway through the import loop releases the peers already
mapped before re-raising. Otherwise a rendezvous that raised would strand a
mapping per peer for the life of the process.

The result is a `_PeerAllocInfo` whose `window` field holds a `_CudaPeerState` —
the imported `_CudaMapping`s plus the multicast object, if any — which is what
`close_peers` later tears down.

## 5. Multicast (multimem)

A multicast object is a separate handle that several devices bind their memory
into. Mapping it yields an address where one store reaches every bound rank's
physical buffer, and a reduce-load returns the reduction across them.

Setting one up is collective, and each step can fail on one rank while
succeeding on the others. The flow agrees with peers before any step that blocks
on them, and publishes success only once the object is mapped:

```
  rank 0: create + export ──>  agree the object exists
  exchange the handle
  import + add_device ──────>  agree before anyone binds
  bind + map ───────────────>  agree before anyone commits
```

Step by step: rank 0 calls `cuMulticastCreate` for `world_size` devices and
exports the handle; the handle travels by whichever transport is in force; every
rank imports it and calls `cuMulticastAddDevice` for its own GPU; each then binds
its physical block into the object with `cuMulticastBindMem` and maps the object
to get the multimem address, offset by `buffer_offset()` to give `multicast_ptr`
at the data buffer.

Three of the agreements are load-bearing:

- **After create *and export*.** Both are inside the guarded block, so a rank 0
  that cannot export degrades multicast for the whole group. If the export sat
  after this vote it would run inside the exchange, and rank 0 raising there
  would leave every peer blocked in a collective it never joins.
- **After `add_device`.** `cuMulticastBindMem` blocks until every device has been
  added. A rank that fails in `add_device` and leaves would wedge all the others
  inside bind — a hang rather than an error, and one no timeout on the collective
  would catch, because they never reach a collective.
- **After `map`.** A bound-but-unmapped object is an intermediate state peers
  cannot use. Publishing success at bind would let a rank whose map failed
  disagree with peers that believe multicast is live, and that divergence only
  surfaces later, at some unrelated collective.

Because multicast is an optimization over the peer-to-peer path, an abandoned
setup is torn down and reported as unavailable rather than raised — callers fall
back to peer pointers. `has_multicast_support(handle)` is the supported test; it
is a property of the allocation, since it depends on the hardware *and* on every
rank having succeeded.

**Access is instruction-restricted.** A multicast address is addressable only
through the `multimem.*` instruction family — ordinary loads and stores are not
the supported path. The kernels live in `torchpyimpl/distributed/symmetric_memory/_multimem.py`.

## 6. Teardown

Order is a correctness constraint:

```
  close_peers     multicast: unmap -> unbind -> release
                  each imported peer mapping: unmap -> release
  free_physical   the local block: unmap -> release
```

`cuMulticastUnbind` must see the bound block still alive, so multicast teardown
precedes the physical release. The allocator base sequences `close_peers` before
`free_physical`, so the backend only keeps its own steps ordered.

Two rules throughout:

- **Synchronize the owning device**, not whichever is current. `free_pool` drains
  outside its `torch.cuda.device` block, so unmapping under the wrong context
  would pull memory out from under kernels still running on the device that owns
  it.
- **Log and continue.** Teardown runs on paths that are frequently already
  failing; stopping at the first driver error would leak everything after it.

Exchanged fds are not part of this: they are closed as soon as the import that
needed them is done (§3), not held for the life of the mapping.

## 7. Hooks

What this backend supplies to `SymmetricMemoryAllocatorBase`
([`allocator.md`](allocator.md) §7):

| Hook | Implementation |
|---|---|
| `alloc_physical` | `cuMemCreate` + reserve + map RW, returning a `_CudaMapping` |
| `free_physical` | synchronize the owning device, then unmap + release |
| `allocation_base` | `cuMemGetAddressRange` |
| `establish_peers` | gather payloads, validate, exchange handles, import + map each peer, then multicast |
| `close_peers` | tear down multicast, then unmap every imported peer mapping |

## 8. Invariants

1. The shareable-handle type is fixed at `cuMemCreate`, and the transport is
   frozen for the process. Ranks that resolved different transports cannot
   interoperate, and are rejected at `_validate` rather than left to deadlock.
2. The granularity-rounded `mapped_size` travels in the payload — every rank maps
   exactly what the exporter created.
3. One rank per GPU, enforced on `(host, device)` at rendezvous. On the fd
   transport the group must also be single-host.
4. Rendezvous happens on the device that owns the allocation.
5. Nothing that can fail on one rank alone precedes a collective without a vote
   in between.
6. Every fd obtained from an exchange is closed once imported.
7. Multicast success is published only after mapping, and only unanimously; a
   partial setup is torn down and reported unavailable.
8. Multicast unbind precedes the physical release of the block it is bound to.
