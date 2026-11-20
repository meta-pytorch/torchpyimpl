# Signal-pad synchronization

`torchpyimpl/distributed/symmetric_memory/_signal.py` — the CuTeDSL kernels behind the `SymmMemHandle` methods `barrier`, `put_signal` and
`wait_signal`.

See [`api.md`](api.md) for the calling surface and
[`allocator.md`](allocator.md) for where the pad sits in an allocation.

## Slot layout

Every allocation carries a signal pad in front of its data buffer — an array of
`uint32`. The slot rank `src` uses to signal rank `dst` on `channel` is:

```
  dst_pad[channel * world_size + src]
```

Channels keep independent synchronizations — different kernels, or different
stages of one kernel — off each other's slots.

The format and offset are the same ones torch's `CUDASymmetricMemory` uses, so
torch's kernels and these can synchronize on one pad within a buffer's lifetime.

## The handshake

Each slot is a two-state flag with exactly one writer per transition:

```
        send (release)                  wait (acquire)
   spin: CAS(slot, 0 -> 1)          spin: CAS(slot, 1 -> 0)
   until it observes 0              until it observes 1
```

Two properties fall out of that shape:

- **Self-resetting.** `wait` leaves the slot at 0, so nothing clears it between
  uses. The allocator's one-time zeroing at alloc is all the initialization
  there is.
- **A fast sender can't overrun a slow receiver.** Send spins until the slot is
  0, so signalling twice before the peer consumes the first blocks rather than
  overwriting.

`barrier` is this applied all-to-all: thread `t` (one per peer) signals peer
`t`, then waits for peer `t`. No send depends on a wait, so the barrier cannot
deadlock.

## Memory ordering

The flag gates other memory traffic — a rank signals to say "what I wrote is
visible to you" — so the qualifiers are the load-bearing part:

- **release** on send: data writes can't sink below the flag.
- **acquire** on wait: subsequent reads can't be hoisted above it.
- **system scope** on both: the ordering has to hold across GPUs. A
  device-scoped atomic compiles, runs, and is subtly wrong.

## Implementation

`cute.arch.atomic_cas` takes `sem` and `scope` as arguments, so the qualifiers
above are part of the call and emit `atom.global.{release,acquire}.sys.cas.b32`
directly — no inline PTX. Each spin is a CAS loop with a `%globaltimer`
deadline.

Pointer chasing (`signal_pad_ptrs[i]` → that rank's pad) happens inside the
kernel via `cute.make_ptr`, which is only legal in a JIT context.

**Compile caching is a requirement, not an optimization.** CuTeDSL JITs on first
call, and for kernels this small a JIT dwarfs everything else. The launchers are
memoized on `(world_size, channel, threads, bounded, device)`, so each shape
pays once per process.

`tests/distributed/symmetric_memory/test_numerical.py` drives a Triton one-shot
all-reduce over symmetric buffers — the memory is ordinary, so any kernel
language can consume it.

## Multicast barrier

When the allocation has a multicast signal pad, `barrier()` uses the multimem
barrier in `_multimem.py` instead of the CAS barrier: one
`multimem.red.release.sys.add` on the channel's counter reaches every rank's
barrier state, each rank waits with an acquire load until its copy reaches
`world_size`, then subtracts `world_size`. One atomic replaces the CAS barrier's
`world_size - 1` round trips.

The counters are in the barrier state (`state[channel]`, see
[`allocator.md`](allocator.md) §6), not in the signal pad, as in torch. In the
pad, the counter slot is also a `put_signal` slot, so a pending signal would
count as an arrival, and a later `wait_signal` would never see its value.

## Behaviour

- **Bounded spins, defaulting to the process group's timeout.** These kernels
  are not process-group work, so the NCCL watchdog never sees them — it times out
  `WorkNCCL` entries and a bare launch creates none, and its remedy
  (`ncclCommAbort`) cannot touch a thread spinning on a raw peer pointer anyway.
  Unbounded, a missing peer wedges the device until the process is killed. So
  `timeout_ms=None` (the default) uses the handle's group timeout (unbounded if
  the group's backend exposes none, or once the group is destroyed); `0` is
  unbounded, as in torch. A thread
  past the deadline traps — an
  unrecoverable CUDA error on the host's next sync, with the pad left
  mid-handshake. The deadline uses `%globaltimer` (real nanoseconds) rather than
  `clock64` (SM cycles, which drift with throttling).
- **Host-side validation.** Peer and channel ranges are checked before launch —
  an out-of-range slot would otherwise be an out-of-bounds atomic on another
  rank's pad, or a spin on a slot nobody will signal.
- **Stream-ordered.** All three are ordinary launches on the current stream, so
  they order against surrounding work like any kernel and are safe to capture in
  a CUDA graph. Rendezvous is not — it is collective and allocates.
- **No communicator involvement.** Peer pads are plain device pointers, so this
  works the same on the CUDA-IPC and NCCL backends.
