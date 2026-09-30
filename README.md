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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

torchpyimpl is BSD 3-Clause licensed, as found in the [LICENSE](LICENSE) file.
