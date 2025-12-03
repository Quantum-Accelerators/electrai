# CHGCAR to Zarr Conversion

This module provides tools to convert Materials Project CHGCAR charge density data from compressed JSON format to Zarr format for efficient storage and access.

## Installation

Install the optional `zarr_conversion` extra that is defined in `pyproject.toml`:

```bash
uv pip install -e ".[zarr_conversion]"
```

## Usage

### Convert a Single File

```bash
uv run python convert_to_zarr.py convert <input.json.gz> <output.zarr>
```

Example:
```bash
uv run python convert_to_zarr.py convert mp-1790998.json.gz mp-1790998.zarr
```

### Convert a Directory of Files

```bash
uv run python convert_to_zarr.py convert_dir <input_dir> <output_dir>
```

Example:
```bash
uv run python convert_to_zarr.py convert_dir ../chgcars ./zarr_output
```

### Convert with Custom Pattern

```bash
uv run python convert_to_zarr.py convert_dir <input_dir> <output_dir> --pattern "mp-*.json.gz"
```

### Parallel Processing

The directory conversion uses parallel processing by default. You can control the number of workers:

```bash
# Use 8 parallel workers
uv run python convert_to_zarr.py convert_dir ../chgcars ./zarr_output --max_workers=8

# Use all available CPU cores (default)
uv run python convert_to_zarr.py convert_dir ../chgcars ./zarr_output
```

## Zarr Structure

Each converted Zarr store contains:

- `charge_density_total/` - 3D array of total charge density (float32)
- `charge_density_diff/` - 3D array of charge density difference for spin-polarized calculations (float32)
- Metadata attributes:
  - `structure` - JSON string containing pymatgen structure information
  - `metadata` - JSON string with task_id and version information
