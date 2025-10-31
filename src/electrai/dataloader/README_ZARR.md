# Zarr Dataloader Documentation

## Overview

The `zarr_s3_read.py` module provides a dataloader for CHGCAR charge density data stored in Zarr format, supporting both local filesystem and S3 storage.

## Key Features

1. **S3 and Local Support**: Seamlessly read from local directories or S3 buckets
2. **Lazy Loading Architecture**: Data is loaded only when needed, not all at once
3. **Drop-in Replacement**: Compatible with existing `RhoData` dataset interface
4. **Registry Integration**: Registered as `"zarr_s3_data"` dataset type

## Components

### `ZarrS3Reader` Class

Main reader class that handles loading zarr stores from various sources.

**Key Methods:**

- `__init__(...)`: Initialize with data directories, functional type, and S3 config
- `read_zarr_store(zarr_path)`: Open a zarr store (lazy - doesn't load data)
- `read_charge_density(zarr_store, density_type)`: Extract charge density arrays
- `get_metadata(zarr_store)`: Extract structure and task metadata
- `data_split()`: Load and split data into train/test sets

### Usage Example

```python
from electrai.dataloader import get_dataset
from electrai.dataloader.dataset import RhoData
from types import SimpleNamespace
import yaml

# Load config
with open("config_zarr.yaml") as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

# Get train/test splits
train_sets, test_sets = get_dataset(cfg)

# Create datasets
train_data = RhoData(
    *train_sets,
    downsample_data=cfg.downsample_data,
    downsample_label=cfg.downsample_label,
    data_augmentation=True,
)

test_data = RhoData(
    *test_sets,
    downsample_data=cfg.downsample_data,
    downsample_label=cfg.downsample_label,
    data_augmentation=False,
)
```

## Configuration

### Local Zarr Files

```yaml
dataset_name: "zarr_s3_data"
data_dir: ./data/MP/zarr_output
label_dir: ./data/MP/zarr_output
map_dir: ./data/MP/map/map_sample.json.gz
functional: GGA
density_type: total  # or 'diff' for spin-polarized
normalize_data: True
train_fraction: 0.8
random_state: 42
```

### S3 Zarr Files

```yaml
dataset_name: "zarr_s3_data"
data_dir: s3://my-bucket/chgcar-data/zarr
label_dir: s3://my-bucket/chgcar-data/zarr
map_dir: ./data/MP/map/map_sample.json.gz
functional: GGA
density_type: total
normalize_data: True
train_fraction: 0.8
random_state: 42

# S3 authentication (optional)
s3_kwargs:
  anon: False  # Set to True for public buckets
  profile: default  # AWS profile name
  # Or use explicit credentials:
  # key: YOUR_ACCESS_KEY
  # secret: YOUR_SECRET_KEY
```

## Zarr Store Structure

Each zarr store (e.g., `mp-12345.zarr/`) should contain:

```
mp-12345.zarr/
├── charge_density_total    # 3D array (float32, chunked)
├── charge_density_diff     # 3D array (optional, for spin-polarized)
└── .zattrs                 # JSON attributes
    ├── structure           # Crystal structure data
    └── metadata            # Task metadata (task_id, fs_id, etc.)
```

This structure is created by the `convert_to_zarr.py` utility.

## Dependencies

**Required:**
- `numpy`
- `zarr`
- `scikit-learn`
- `monty`

**Optional (for S3 support):**
- `s3fs` - Install with: `pip install s3fs`

## Advantages Over JSON.gz Format

1. **Memory Efficiency**: Zarr stores support chunked, compressed arrays that can be read partially
2. **Fast Access**: Direct array access without full decompression
3. **Cloud Native**: Efficient S3 integration with range requests
4. **Scalability**: Handle datasets that don't fit in memory
5. **Metadata**: Structured storage of structure and task information

## Migration from chgcar_read.py

The `ZarrS3Reader` maintains the same interface as `RhoRead`:

| Feature | chgcar_read.py | zarr_s3_read.py |
|---------|----------------|-----------------|
| Input format | JSON.gz | Zarr |
| Memory loading | All at once | All at once* |
| S3 support | No | Yes |
| Local support | Yes | Yes |
| Train/test split | Yes | Yes |
| Registry integration | Yes | Yes |

*Note: Current implementation loads all data for compatibility with `RhoData`. For true lazy loading, a new `ZarrDataset` class would be needed (future enhancement).

## Converting Existing Data

Use the `convert_to_zarr.py` utility:

```bash
uv run python src/electrai/data/MP/zarr_conversion/convert_to_zarr.py convert_dir \
  --input-dir ./data/MP/chgcars \
  --output-dir ./data/MP/zarr_output \
  --pattern "*.json.gz"
```

## Future Enhancements

1. **True Lazy Loading**: Create `ZarrDataset` class that loads data in `__getitem__`
2. **Caching**: Add in-memory cache for frequently accessed samples
3. **Parallel Loading**: Use multiprocessing for faster data loading
4. **On-the-fly Augmentation**: Apply augmentation during zarr read
5. **Compression Options**: Support different compression codecs (blosc, gzip, etc.)
