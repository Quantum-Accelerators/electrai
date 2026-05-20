#!/bin/bash
#SBATCH --job-name=c3n_preprocess
#SBATCH --output=inference_benchmark/logs/slurm_%j.out
#SBATCH --error=inference_benchmark/logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=64G

# Sample test-set materials from rho_gga and convert each zarr into
# ChargE3Net's .npy + _atoms.pkl format. CPU-only, fast queue. Override
# INPUT/SPLIT/LIMIT via --export if needed.

module load anaconda3/2025.12
conda activate ${HOME}/.conda/envs/charge3net

cd /scratch/gpfs/ROSENGROUP/${USER}/charge3net-benchmark

mkdir -p inference_benchmark/logs

DATASET=${DATASET:-/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/rho_gga}
INPUT=${INPUT:-${DATASET}/label}
FILELIST=${FILELIST:-${DATASET}/mp_filelist.txt}
SPLIT=${SPLIT:-${DATASET}/split_limit_22M.json}
OUTPUT=${OUTPUT:-/scratch/gpfs/ROSENGROUP/${USER}/charge3net-benchmark/data_preprocessed_rho_gga}
LIMIT=${LIMIT:-1000}
WORKERS=${WORKERS:-16}

echo "Dataset:  ${DATASET}"
echo "Input:    ${INPUT}"
echo "Filelist: ${FILELIST}"
echo "Split:    ${SPLIT}"
echo "Output:   ${OUTPUT}"
echo "Limit:    ${LIMIT} (test partition only)"
echo "Workers:  ${WORKERS}"

srun python inference_benchmark/preprocess_density.py \
    --input "${INPUT}" \
    --filelist "${FILELIST}" \
    --split-file "${SPLIT}" \
    --output "${OUTPUT}" \
    --limit "${LIMIT}" \
    --workers "${WORKERS}"
