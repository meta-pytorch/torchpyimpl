# Symmetric memory design notes

Read in this order; each builds on the one before:

| Doc | Covers |
|---|---|
| [`allocator.md`](allocator.md) | The Python allocator on a `torch.cuda.MemPool` / `PrivatePool`, the backend-agnostic base class and its hooks, the signal-pad-first layout, records and caches. |
| [`api.md`](api.md) | The public surface, its parity with `torch.distributed._symmetric_memory`, and the handle accessor contract. |
| [`cuda_ipc.md`](cuda_ipc.md) | The `"CUDA"` backend: CUDA VMM allocation, the fabric and POSIX-fd transports, rendezvous votes, and multicast setup. |
| [`nccl.md`](nccl.md) | The `"NCCL"` backend: `ncclMemAlloc`, symmetric window registration on the process group's communicator, and peer/multimem pointers. |
| [`signal.md`](signal.md) | The signal-pad protocol: slot layout, the CAS handshake and its memory ordering, bounded spins, and the multicast barrier. |
