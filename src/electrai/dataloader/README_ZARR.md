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

### Usage Example

```python
from electrai.dataloader import get_dataset
from electrai.dataloader.dataset import RhoData
from types import SimpleNamespace
import yaml

# Load config
with open("config_zarr.yaml") as f:  # Specify in the configuration to use zarr
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
    └── metadata            # Task metadata (task_id, version, etc.)
```

This structure is created by the `convert_to_zarr.py` utility.

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
