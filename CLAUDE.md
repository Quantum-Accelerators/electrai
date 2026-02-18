ElectrAI is a PyTorch based super-resolution machine learning model for materials. Currently, it implements a ResNet neural network architecture using PyTorch Lightning.

Begin with example training code in examples/MP/run.sh.

Always use uv when running python code and managing its environment.

## Environment Setup

```bash
# Install dependencies with uv
uv sync
```

## Code Quality Checks

```bash
# Run pre-commit hooks (linting, formatting)
uvx pre-commit run --all-files

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/electrai
```

## Training

Please ask if you need to train a model.

## Configuration

Training is controlled via YAML config files in `src/electrai/configs/`. Key parameters:
- `n_channels`, `n_residual_blocks`: Model architecture
- `lr`, `epochs`, `nbatch`: Training hyperparameters
- `warmup_length`: Learning rate warmup epochs
- `wandb_mode`: "online", "offline", or "disabled"

## Data Formats

Supports multiple data sources:
- Native CHGCAR files (via pymatgen) **Currently preferred**
- Compressed JSON (.json.gz)
- Zarr format (local or S3 paths)

Use `zarr_conversion/` utilities to convert between formats.

## Project Structure

```text
src/electrai/
├── lightning.py          # PyTorch Lightning training module
├── configs/              # YAML training configs
├── dataloader/           # Data loading (mp.py, mp_zarr_s3_data.py, registry.py)
├── entrypoints/          # CLI entry points (train.py, main.py)
├── loggers/              # Logging utilities
├── model/                # Neural network (srgan_layernorm_pbc.py) and losses
└── zarr_conversion/      # Data format conversion tools

data/                     # Example data
examples/                 # Entry to running training
tests/                    # pytest test suite
```

Key files:
- `lightning.py` - LightningGenerator class with train/val steps
- `model/srgan_layernorm_pbc.py` - GeneratorResNet architecture
- `dataloader/mp.py` - Materials Project dataset loader
- `dataloader/registry.py` - Dataset registration via @register_data decorator
- `entrypoints/train.py` - Training orchestration
