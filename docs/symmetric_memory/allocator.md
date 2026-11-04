# Symmetric memory allocator design

Covers the two lower layers of the stack:

| Layer | Files |
|---|---|
| driver primitives | `torchpyimpl/distributed/symmetric_memory/_cuda.py` |
| symmetric-memory allocator | `torchpyimpl/distributed/symmetric_memory/_symm_mem.py`, `torchpyimpl/distributed/symmetric_memory/_allocator.py`, `torchpyimpl/distributed/symmetric_memory/_mempool.py` |

This layer is backend-agnostic. The backends that plug into it are documented
separately: [`cuda_ipc.md`](cuda_ipc.md) and [`nccl.md`](nccl.md). The
torch-shaped public API is [`api.md`](api.md).

## 1. The problem

Symmetric memory needs every rank's buffer to be addressable from every other
rank. That requires two things ordinary tensors don't:

1. **Special physical memory.** The block has to be allocated so that it can be
   shared — exportable, or registerable with the communication library — which
   `cudaMalloc` memory is not.
2. **A collective step to publish it.** However peers are reached, every rank
   must take part, with matching allocations in the same order.

The naive approach — allocate and publish per tensor — is unworkable on both
counts. Device-level allocation synchronizes the device, the shareable kinds are
worse, and a collective on every allocation would serialize the whole job. The
whole point of PyTorch's CUDA caching allocator (CCA) is to avoid exactly this:
grab large segments once, hand out blocks from its own free lists, and recycle rather than
return to the driver.

So the real problem is not "how do we allocate symmetric memory" — it's **how do
we get the CCA to do its caching for us while supplying our own physical
memory**, so that a freed symmetric tensor's segment — and, crucially, whatever
cross-rank registration it carries — survives to be reused.

## 2. Where PyTorch lets us in

The CCA has exactly one extension point for this: a **`PrivatePool` with a custom
`allocator_`**.

```
  torch.cuda.MemPool.from_py_allocator(alloc_fn, free_fn,
                                       use_on_oom=False, no_split=True)
        |                                    creates/increfs a PrivatePool
        v
  with torch.cuda.use_mem_pool(pool):        pushes (pool_id, stream filter)
      t = torch.empty(...)                   onto the device's allocation_scopes_
        |
        v
  DeviceCachingAllocator.get_pool()          walks scopes LIFO, routes into
        |                                    the PrivatePool instead of the
        v                                    default small/large pools
  PrivatePool.allocator_  ->  our alloc_fn   only called on a *cache miss*
```

Two properties make this work:

- **A `PrivatePool` has its own free lists** but is otherwise a normal pool — it
  splits, coalesces, recycles and reports stats like any other. We inherit all
  of that for free.
- **Routing into a pool is decoupled from CUDA graph capture.** The CCA tracks
  real captures (`capture_tracker_`) separately from `allocation_scopes_`, so
  entering a pool does not put the allocator into capture mode or trigger
  capture-only rules.

`MemPool.from_py_allocator` accepts Python callables directly:

```python
def alloc_fn(size: int, device: int, stream: int) -> int | None: ...
def free_fn(ptr: int, size: int, device: int, stream: int) -> None: ...
```

The callbacks forward to a `PoolBackend` returned by a resolver registered with
`_allocator.set_backend_resolver()`. `_symm_mem.py` registers
`get_allocator_instance` at import, so the pool always routes to the active
backend. The resolver is called per alloc/free, and PyTorch retains the Python
callables for the allocator's lifetime.

## 3. What the CCA gives us, and what it demands back

The payoff is large, and every constraint below follows from it.

**We get caching of the expensive thing.** With a Python allocator, `free_fn`
does *not* run when a tensor is freed — the block just returns to the pool's free
list. It runs only when the CCA actually releases the segment (`empty_cache`, or
pool teardown). So releasing the physical memory and undoing the cross-rank
registration happen at cache eviction, not per tensor. A training loop that
allocates and frees the same shape every step pays that cost once.

That inverts the usual teardown story, and it is why release is deferred. The
CCA can release a segment on one rank and keep it on another. NCCL window
deregistration is local to the rank, but NCCL gives each new window an offset
that depends on the windows the rank has, and binds multicast at that offset.
If the ranks keep different windows, later windows get different offsets. So
`free_fn` only retires the record onto a queue, and `free_pool`, which every
rank calls, drains that queue once the pool is gone. The drain goes in
**allocation** order (each record carries a `seq` stamped at allocation time),
so it is the same on every rank.

The cost is that a segment the CCA believes it has released is still held by us
until the next drain, so a bare `torch.cuda.empty_cache()` on a symmetric pool
retains that memory until `free_pool`. That is the deliberate trade: windows
that differ across ranks break multicast addressing, which is worse than memory
held to teardown.

**Constraints the CCA imposes:**

- **Callbacks run without the CCA mutex held**, but may not recursively allocate
  from their own pool. PyTorch detects that re-entry and reports it as an error.
  This is why the signal pad is zeroed with `cuMemsetD8` rather than
  `torch.empty().zero_()`.
- **`use_on_oom=False`** — the pool's space must never be lent to a
  non-symmetric allocation. Ranks allocate in lockstep; letting an unrelated
  tensor consume pool space on one rank would desync the allocation sequence and
  the mappings would stop lining up.
- **`no_split=True`** — one allocation per segment. A segment carries exactly one
  signal pad, so two tensors sharing a segment would race on it. It also keeps a
  tensor's `data_ptr` equal to its allocation's data pointer, which is what makes
  `rendezvous`'s exact-match lookup hit.
- **PyTorch owns callback failure handling.** An allocation exception is
  surfaced with its message; a free exception is logged and swallowed.
- **The callbacks receive the allocation device and raw CUDA stream.** A backend
  using stream-ordered operations is responsible for the corresponding device
  and stream context. During graph capture, it must not launch work on or
  synchronize the capturing stream.

**Cross-rank ordering is the one thing we can't inherit.** The CCA stamps each
segment with a monotonic `registration_counter`, and `ProcessGroupNCCL` sorts by
it when registering a `MemPool` with NCCL, so all ranks register in creation
order. That field is not exposed to Python (snapshot blocks carry `address`,
`size`, `requested_size`, `state` and `frames`), so the allocator keeps its own
ordering: `_records` is an insertion-ordered dict, and insertion order is
allocation order, identical across ranks when every rank performs the same
allocations.

## 4. Layering

```
  torch.cuda.MemPool.from_py_allocator  _mempool.py           pool per device
        |
  Python alloc/free callbacks           _allocator.py         PoolBackend resolver
        |
  SymmetricMemoryAllocatorBase          _symm_mem.py          layout + records
        |  (backend hooks)
  a backend                             backends/            physical memory +
        |                                                    cross-rank mapping
  driver / communication library
```

`_symm_mem.py` is backend-agnostic: it imports the backend lazily from
`get_allocator_instance()`, both to break the import cycle (a backend subclasses
the base) and so a process never pays for a backend it doesn't use.

## 5. Primitives

**`_cuda.py`** — CUDA driver calls via `cuda.bindings.driver`, behind an
`lru_cache`'d `cuInit` so importing torchpyimpl does not require a GPU.

- `cu_mem_get_address_range(ptr)` — allocation base for an interior pointer.
  `CUDA_ERROR_NOT_FOUND` means "not a device allocation" and returns `0` (logged
  at debug); every other driver error raises, so a dead context cannot
  masquerade as a lookup miss.
- `cu_memset_d8(ptr, value, n)` — zeroes the signal pad without re-entering the
  torch caching allocator, then synchronizes the context, so no peer can signal
  before the zeroing is done. `n == 0` is a no-op; `n < 0` raises.
- `should_skip_cuda_cleanup(device)` — synchronizes `device` before teardown;
  returns `True` during interpreter shutdown or after a CUDA error, so backends
  defer teardown instead of freeing memory under a poisoned context.

## 6. Memory layout

Signal-pad-first, matching torch's layout:

```
  alloc_base                            buffer_ptr = alloc_base + buffer_offset
  |                                     |
  v                                     v
  [ signal pad    | barrier state     ] [ data buffer | round to 16 ]
  <-- 2 * unit --> <----- unit ------>
  <----------- buffer_offset ---------> <--- round_up(size, 16) --->
```

`unit == round_up(ceil(signal_pad_bytes() / 2), 128)` and
`buffer_offset() == 3 * unit`, as in torch. The barrier state starts at
`barrier_state_offset() == 2 * unit` and holds one multimem barrier counter per
channel, apart from the signal pad (see [`signal.md`](signal.md)). The pad size
freezes on the first allocation. Being a process-global constant is what lets
`find_covering` recover a data pointer from any interior pointer in O(1): ask the
driver for the allocation base, add the constant. Default pad is
`32 * 72 * 4 = 9216` B (torch's `symm_max_nblocks * max_cuda_p2p_domain_size *
sizeof(uint32)`), overridable via `TORCHPYIMPL_SIGNAL_PAD_BYTES` or
`set_signal_pad_size()`.

## 7. Base class and hooks

`SymmetricMemoryAllocatorBase` owns the common allocator state and operations.
These operations are `alloc`, `free`, `find_covering`, `rendezvous`, and
`release_peer_resources`. The base class also owns the record map, the pad
geometry, and the handle cache. A backend supplies only:

| Hook | Responsibility |
|---|---|
| `alloc_physical(block_size, device)` | physical memory → `(base, handle)` |
| `free_physical(handle)` | release it and report cleaned/deferred/failed |
| `allocation_base(ptr)` | interior ptr → allocation base |
| `establish_peers(rec, comm_ptr, world_size, device, group)` | cross-rank mapping |
| `close_peers(pai)` *(optional)* | drop that mapping and report cleanup status |
| `peers_alive(pai)` *(optional)* | `False` if the mapping died with its group; `rendezvous` then builds a new one (default: `True`) |
| `initialize_physical(handle, base, size)` *(optional)* | zero the signal pad (default: `cuMemsetD8`) |
| `should_skip_cleanup(device)` *(optional)* | defer teardown after a CUDA error (default: `should_skip_cuda_cleanup`) |

A backend is a module under `backends/`, plus any helper modules (e.g.
`_ipc_fd`). `establish_peers`
receives both `comm_ptr` and `group`, so a backend can reach peers through the
communicator or through a host collective on the process group, whichever suits
it.

## 8. Records and caches

Three records, nested so that lifetime is automatic:

- **`AllocRecord`** — one physical allocation. Keyed in `_records` by its **data**
  pointer; insertion order is allocation order (see §3).
- **`_PeerAllocInfo`** — one per `(allocation, communicator)`. Owns the single
  cross-rank mapping and the per-peer pointer arrays. Reused by every view into
  the allocation, so a slice at a different offset does not re-register.
- **`SymmMemHandle`** — what `rendezvous` returns. `buffer_ptrs[i]` /
  `signal_pad_ptrs[i]` are peer bases. `buffer_ptr` is `buffer_ptrs[rank]` plus
  the view offset; on NCCL this is a window address, which can differ from the
  tensor's pointer. `data_ptr` is the pointer passed to `rendezvous`.
  `multicast_ptr` is shifted by the view offset.

Both caches hang off the record — `rec.per_comm` and `rec.handles` — rather than
living in process-global maps with reverse indexes. Freeing an allocation drops
the record and therefore every handle and mapping over it, so a freed allocation
can never leave a dangling slice handle behind, for any backend.

## 9. Invariants

1. The pad size is frozen at the first allocation; `buffer_offset` is constant
   for the rest of the process.
2. Establishing a cross-rank mapping is collective for every backend; dropping
   one is local for every backend. Every rank must rendezvous matching
   allocations in the same order. For NCCL, every rank must also drop the same
   windows before it registers a new one on that communicator.
3. `alloc_fn` must not recursively allocate from its own pool. `free_fn` must
   not release mappings, because the CCA can release a segment on one rank only.
   It only retires the record and queues it; `release_pending` closes the
   mappings and frees the memory in allocation order, from a path that every
   rank runs.
4. `find_covering` reports a hit only inside `[buffer_ptr, buffer_ptr +
   buffer_size)`. The signal pad in front and the 16-byte rounding tail belong to
   the physical allocation but to no view.
