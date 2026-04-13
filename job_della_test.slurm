#!/bin/bash
#SBATCH --job-name=della-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/ROSENGROUP/bb9080/logs/della-test-%j.out

module purge
module load anaconda3/2025.6
conda activate electrai
module load proxy/default

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

export TORCH_DIST_INIT_BARRIER=1
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO

cd /scratch/gpfs/ROSENGROUP/bb9080/electrai

export PYTHONPATH=$(pwd)
export PYTORCH_ALLOC_CONF=expandable_segments:True

srun uv run --no-sync torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=$SLURM_GPUS_PER_NODE \
        --rdzv_id=2001 \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        ./src/electrai/entrypoints/main.py train \
        --config ./src/electrai/configs/MP/config_della_test.yaml
