#!/bin/bash
#SBATCH --job-name=c3n_throughput
#SBATCH --output=inference_benchmark/logs/slurm_%j.out
#SBATCH --error=inference_benchmark/logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --constraint=gpu80
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# Headline run: 1 node, 1 GPU. This is the production-like single-GPU
# throughput we want to compare against ResUNet. To finish faster on the
# full test split, raise --gres and NPROCS to shard the data — per-rank
# throughput still represents single-GPU performance.

module load anaconda3/2025.12
conda activate ${HOME}/.conda/envs/charge3net

cd /scratch/gpfs/ROSENGROUP/${USER}/charge3net-benchmark

mkdir -p inference_benchmark/logs

export NNODES=1
export NPROCS=1

# Unbuffered stdout so progress lands in the .out file in real time
# (otherwise a SLURM timeout truncates everything).
export PYTHONUNBUFFERED=1

srun python -u inference_benchmark/launch_benchmark.py
