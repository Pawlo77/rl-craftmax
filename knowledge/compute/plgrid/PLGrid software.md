---
title: PLGrid software
type: handbook
domain: plgrid
tags: [handbook, plgrid, modules]
created: 2026-10-11
updated: 2026-10-11
parent: "[[PLGrid compute]]"
---
# PLGrid software

Snapshot of the Helios GPU module list of 2026-10-11; the list is rebuilt daily.

## Lmod

Software is provided as modules loaded dynamically:

| Command | Effect |
|---------|--------|
| `module avail` | List available modules |
| `module list` | Show loaded modules |
| `module load <name>/<version>` | Load a module; many need their toolchain first, for example `GCCcore/14.3.0 Python/3.12.13` |
| `module unload <name>` | Unload one module |
| `module purge` | Unload all modules |

Missing software is requested through the Helpdesk with its name, version and source. Some packages need their own licence and are not open to all users.

## Helios GPU highlights

343 module families are listed. The ones relevant to Python and GPU work:

| Family | Versions |
|--------|----------|
| CUDA | 12.4.0, 12.8.0, 12.9.1, 13.0.2, 13.2.0 |
| cuDNN | 8.9 and 9.2 to 9.23, each tied to a CUDA version |
| NCCL and `aws-ofi-nccl` | 2.25 to 2.31 (the latter also tied to CUDA 12.8 to 13.2) |
| Python | 3.11.5, 3.12.13, 3.13.5, 3.14.2 |
| Python bundles | `Python-bundle-PyPI` 2023.10, 2025.07, 2026.04; `SciPy-bundle` 2023.11, 2025.07; `ML-bundle` 24.06a, 25.04, 25.10; `matplotlib`, `mpi4py`, `PyYAML`, `pydantic`, `SQLAlchemy`, `psutil` |
| Environment tools | `virtualenv`, `poetry`, `git`, `git-lfs`, `rclone`, `MinIO-Client`, `tmux`, `nvtop`, `parallel`, `Rust`, `Go` |
| Compilers and MPI | GCC 13 to 15, NVHPC, LLVM, OpenMPI 5.0.x, UCX |
| Interactive | JupyterLab, Jupyter-bundle, VSCode, `pro-viz` |

There is no standalone PyTorch family in the list; PyTorch would come from `ML-bundle` or a virtual environment. Check the bundle contents before relying on either (<https://kompendium.plgrid.pl/>).

## Choosing versions

- Module names encode their toolchain and CUDA version (for example `cuDNN/<version>-CUDA-12.9.1`); load libraries that name the same CUDA version together.
- A self-built environment must be built for the architecture of the compute nodes.
