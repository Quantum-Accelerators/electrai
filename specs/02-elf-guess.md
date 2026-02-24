# Elf-Guess: Initial Guess Library

## Overview

Hananeh is building a library for generating initial guesses (SAD — Superposition of Atomic Densities) for CHGCAR and ELFCAR files. This will be packaged as a separate Python project within the same repo, using a **uv workspace** (monorepo with multiple packages).

## Background

The ML model takes a pair of files per sample:
- **Input**: cheap/approximate density (SAD guess or coarse DFT)
- **Label**: converged DFT density (expensive to compute)

Currently, input generation (via QuAcc / pymatgen) is ad hoc. This library would standardize and automate it.

For CHGCARs, the SAD approach sums spherical atomic densities. For ELFCARs, Hananeh found **GPAW** (a DFT code) can produce ELF initial guesses.

## Architecture

```
electrai/                          # repo root
├── pyproject.toml                 # root workspace config
├── src/electrai/                  # existing ML model package
├── packages/
│   └── elf-guess/                 # new package
│       ├── pyproject.toml
│       ├── src/elf_guess/
│       │   ├── __init__.py
│       │   ├── sad.py             # SAD guess generation
│       │   ├── gpaw_elf.py        # GPAW-based ELF guess
│       │   └── cli.py             # click CLI
│       └── tests/
└── uv.lock                       # shared lockfile
```

### uv Workspace Setup

Root `pyproject.toml` addition:
```toml
[tool.uv.workspace]
members = ["packages/*"]
```

`packages/elf-guess/pyproject.toml`:
```toml
[project]
name = "elf-guess"
dependencies = ["pymatgen", "gpaw", ...]
```

The main `electrai` package can optionally depend on `elf-guess`:
```toml
[project.optional-dependencies]
guess = ["elf-guess"]
```

## Tasks

- [ ] Wait for Hananeh to push her initial guess code
- [ ] Set up uv workspace in repo root
- [ ] Create `packages/elf-guess/` package structure
- [ ] Integrate with existing dataloader (generate inputs on-the-fly?)
- [ ] Add CLI entry point: `elf-guess generate --structure POSCAR --output input.CHGCAR`
- [ ] Tests: verify generated guesses match expected SAD output
- [ ] CI: add `elf-guess` to test matrix

## Open Questions

- Does GPAW require a specific Python version or system dependencies?
- Should initial guess generation be a build-time step (pre-training) or runtime (on-the-fly in dataloader)?
- How does this interact with the S3 data pipeline? Generate once and upload, or generate per-run?
