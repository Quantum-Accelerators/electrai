#!/bin/bash

cd ../../

export PYTHONPATH=$(pwd)
export PYTORCH_ALLOC_CONF=expandable_segments:True
uv run ./src/electrai/entrypoints/main.py train --config  ./src/electrai/configs/MP/config.yaml
