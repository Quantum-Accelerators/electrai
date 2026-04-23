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

Each converted Zarr store contains the full CHGCAR content. Total and diff
densities are stored as independent zarr arrays so they can be chunked and
loaded separately (training typically reads one or the other, not both).

Arrays:

- `charge_density_total/` - 3D float32 total charge density
- `charge_density_diff/` - 3D float32 magnetization density (spin-polarized only)
- `charge_density_diff_x/`, `charge_density_diff_y/`, `charge_density_diff_z/` -
  non-collinear magnetization components (SOC calculations only)

Attributes:

- `structure` - JSON pymatgen Structure
- `metadata` - JSON with `task_id`, `pymatgen_version`
- `data_aug` - JSON dict of PAW augmentation occupancy lines, keyed by
  density component (`total`, `diff`, ...)
- `poscar_comment` - POSCAR header/comment string (may be null)
- `is_spin_polarized`, `is_soc` - bool flags

### Chunking

Pass `--chunks` and `--chunks_diff` to control chunk sizes independently for
total and diff arrays:

```bash
uv run python convert_to_zarr.py convert input.CHGCAR output.zarr \
  --chunks "(32,32,32)" --chunks_diff "(16,16,16)"
```

`chunks_diff` defaults to `chunks` when not provided.

Pass `--write_diff=False` to skip all diff arrays (total-only output):

```bash
uv run python convert_to_zarr.py convert input.CHGCAR output.zarr --write_diff=False
```
