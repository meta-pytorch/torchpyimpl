<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/torch-pyimpl-logo-dark.png">
    <img width="55%" src="docs/assets/torch-pyimpl-logo.png" alt="torchpyimpl">
  </picture>
</p>

# torchpyimpl

torchpyimpl provides pure-Python implementations of PyTorch GPU and distributed
subsystems, built on vendor Python GPU libraries (`cuda-python`, `nccl4py`,
CuTeDSL, Triton). The implementations are API-compatible with PyTorch, operate
on regular `torch.Tensor`s, and require no C++ build.

> [!WARNING]
> torchpyimpl is experimental and under active development. APIs may change.

## Components

| Component | Module | Status |
|---|---|---|
| Symmetric memory | `torchpyimpl.distributed.symmetric_memory` | In progress |

Design notes: [`docs/symmetric_memory`](docs/symmetric_memory/README.md).

## Prerequisites

- Python 3.12 through 3.14
- PyTorch nightly
- CUDA 13 capable NVIDIA GPU

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/meta-pytorch/torchpyimpl.git
cd torchpyimpl
uv sync
```

With pip, install PyTorch nightly first:

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu132
pip install -e .
pip install --group dev    # pip >= 25.1
```

## Development

Lint and format with [lintrunner](https://github.com/suo/lintrunner):

```bash
uv run lintrunner init
uv run lintrunner          # lint changed files
uv run lintrunner -a       # apply formatting fixes
```

Run tests:

```bash
uv run pytest tests/                    # GPU tests are skipped without GPUs
uv run pytest tests/ -m "not gpu and not multi_gpu"
```

## Benchmarks

`benchmarks/symmetric_memory/bench.py` times allocation, cold rendezvous
(allocation + cross-rank setup) and cached rendezvous for one backend, and
optionally torch's native symmetric memory for the same backend. Results are
p20/p50/p80 wall-clock per buffer size; the table shows rank 0 and the raw
per-rank JSON is written under `$TORCHPYIMPL_BENCH_DIR` (default
`/tmp/torchpyimpl-bench`).

```bash
# One configuration (default: CUDA backend, torchpyimpl, cached rendezvous, all GPUs)
uv run python benchmarks/symmetric_memory/bench.py
uv run python benchmarks/symmetric_memory/bench.py --backend NCCL --with-alloc-sweeps
uv run python benchmarks/symmetric_memory/bench.py --backend CUDA --measure torch
uv run python benchmarks/symmetric_memory/bench.py --world-size 4 --min-log2 12 --max-log2 20

# All four configurations (torch CUDA, torchpyimpl CUDA, torchpyimpl NCCL,
# torch NCCL), each in its own process
./benchmarks/symmetric_memory/compare.sh --with-alloc-sweeps
RANKS="2 4 8" ./benchmarks/symmetric_memory/compare.sh      # sweep world size
```

| flag | meaning |
|---|---|
| `--backend CUDA\|NCCL` | symmetric-memory backend (default `CUDA`) |
| `--measure ours\|torch\|both` | time torchpyimpl, torch's native implementation, or both in one process (default `ours`) |
| `--with-alloc-sweeps` | also run the allocation and cold-rendezvous sweeps |
| `--world-size N` | ranks to use (default: all visible GPUs, up to 8) |
| `--min-log2` / `--max-log2` | buffer sizes as powers of two (default 4 KiB to 1 GiB) |

`compare.sh` runs one process per configuration because torch's NCCL backend and
the torchpyimpl NCCL backend register windows on the same communicator and would
contend if measured together. Extra arguments are passed to every `bench.py`
run.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Third-party dependencies

torchpyimpl requires the third-party packages below. They are not part of
torchpyimpl and are not distributed with it; your package manager installs
them from their own sources. Each package is subject to its own license terms,
which are separate from the torchpyimpl license. Review and follow those terms
when you install or use them.

| Package | Required for | License |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) (`torch`) | runtime | BSD 3-Clause; includes third-party components under other licenses |
| [Triton](https://github.com/triton-lang/triton) (`triton`, Linux) | runtime | MIT |
| [CUDA Python](https://github.com/NVIDIA/cuda-python) (`cuda-python`) | runtime | Apache 2.0 |
| [nccl4py](https://github.com/NVIDIA/nccl) (`nccl4py`) | runtime | Apache 2.0 |
| [CUTLASS DSL / CuTeDSL](https://github.com/NVIDIA/cutlass) (`nvidia-cutlass-dsl`) | runtime | [NVIDIA Software License Agreement](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html) |
| [pytest](https://github.com/pytest-dev/pytest) (`pytest`) | development | MIT |
| [lintrunner](https://github.com/suo/lintrunner) (`lintrunner`) | development | MIT |

These packages also install their own dependencies. These include NVIDIA CUDA
libraries (for example `nvidia-nccl-cu13` and other `nvidia-*` packages), which
are subject to NVIDIA license terms. The license for each package is in its
package metadata and its source repository.

## License

torchpyimpl is BSD 3-Clause licensed, as found in the [LICENSE](LICENSE) file.
