#!/bin/bash
cd ../../

export PYTHONPATH=$(pwd)

python3 ./src/electrai/entrypoints/main.py train --config  ./src/electrai/configs/QM9/config.yaml
