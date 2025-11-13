#!/bin/bash

cd ../../

export PYTHONPATH=$(pwd)
export PYTORCH_ALLOC_CONF=expandable_segments:True
python3 ./src/electrai/entrypoints/main.py train --config  ./src/electrai/configs/MP/config.yaml
