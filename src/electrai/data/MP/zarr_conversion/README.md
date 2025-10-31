# CHGCAR to Zarr Conversion

This module provides tools to convert Materials Project CHGCAR charge density data from compressed JSON format to Zarr format for efficient storage and access.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv pip install -r requirements.txt
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

## Zarr Structure

Each converted Zarr store contains:

- `charge_density_total/` - 3D array of total charge density (float32)
- `charge_density_diff/` - 3D array of charge density difference for spin-polarized calculations (float32)
- Metadata attributes:
  - `structure` - JSON string containing pymatgen structure information
  - `metadata` - JSON string with task_id, fs_id, and version information

## Benefits of Zarr Format

1. **Efficient Compression**: ~45% size reduction compared to JSON.gz
2. **Chunked Access**: Read specific regions without loading entire array
3. **Parallel I/O**: Multiple processes can read simultaneously
4. **Cloud-Ready**: Works with cloud storage backends (S3, GCS, etc.)
5. **Interoperability**: Compatible with Dask, Xarray, and other scientific Python tools

## Performance

Tested on 10 CHGCAR files:
- Original JSON.gz: 92M
- Zarr output: 51M
- Compression improvement: ~45% reduction
- All conversions completed successfully
