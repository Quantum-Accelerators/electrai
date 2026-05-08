#!/bin/bash
#SBATCH --job-name=resunet_throughput
#SBATCH --output=/scratch/gpfs/ROSENGROUP/%u/logs/resunet-throughput-%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/%u/logs/resunet-throughput-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --constraint=gpu80
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# Matched ResUNet3D throughput benchmark. Runs the same materials used by
# the charge3net benchmark and writes a CSV with the same per-material
# schema, so the two CSVs can be joined on filename.
#
# Defaults assume:
#   - charge3net-benchmark/data_preprocessed/filelist.txt holds the
#     matched mpids (sampled from rho_gga's test partition)
#   - chg_datasets/rho_gga/data has <mpid>.zarr files
#   - ckpt_1 is the rho_gga-trained checkpoint, so the test partition is
#     a true held-out set (ckpt_2 is rho_gga+u-trained — wrong functional)
#
# Override any of these with --export=ALL,VAR=value at sbatch time. Use
# SHAKEDOWN=20 first to verify the run is working before going full size.

module purge
module load anaconda3/2025.6
conda activate electrai
module load proxy/default

cd /scratch/gpfs/ROSENGROUP/${USER}/electrai2

mkdir -p /scratch/gpfs/ROSENGROUP/${USER}/logs

CKPT=${CKPT:-/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_checkpoints/04_22_2026/ckpt_1/ckpt_epoch=30_val_loss=0.010749.ckpt}
FILELIST=${FILELIST:-/scratch/gpfs/ROSENGROUP/${USER}/charge3net-benchmark/data_preprocessed_rho_gga/filelist.txt}
INPUT_DIR=${INPUT_DIR:-/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/rho_gga/data}
OUTPUT=${OUTPUT:-/scratch/gpfs/ROSENGROUP/${USER}/charge3net-benchmark/resunet_results}
SHAKEDOWN=${SHAKEDOWN:-20}

# Architecture knobs — must match the checkpoint's training config.
# Defaults match configs/MP/config_resunet_baseline.yaml.
N_CHANNELS=${N_CHANNELS:-32}
N_RES_BLOCKS=${N_RES_BLOCKS:-1}
KERNEL_SIZE=${KERNEL_SIZE:-5}
DEPTH=${DEPTH:-2}

echo "Checkpoint:    ${CKPT}"
echo "Filelist:      ${FILELIST}"
echo "Input dir:     ${INPUT_DIR}"
echo "Output dir:    ${OUTPUT}"
echo "Shakedown:     ${SHAKEDOWN}"
echo "Architecture:  n_channels=${N_CHANNELS} blocks=${N_RES_BLOCKS} kernel=${KERNEL_SIZE} depth=${DEPTH}"
echo "Start:         $(date)"
echo

# Unbuffered stdout so progress lands in the .out file in real time.
export PYTHONUNBUFFERED=1

srun uv run --no-sync python -u scripts/benchmark_resunet_throughput.py \
    --checkpoint "${CKPT}" \
    --filelist "${FILELIST}" \
    --input-dir "${INPUT_DIR}" \
    --output "${OUTPUT}" \
    --shakedown-limit "${SHAKEDOWN}" \
    --n-channels "${N_CHANNELS}" \
    --n-residual-blocks "${N_RES_BLOCKS}" \
    --kernel-size "${KERNEL_SIZE}" \
    --depth "${DEPTH}"

echo
echo "End:           $(date)"
