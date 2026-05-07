#!/bin/bash
#SBATCH --job-name=c3n_preprocess
#SBATCH --output=inference_benchmark/logs/slurm_%j.out
#SBATCH --error=inference_benchmark/logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=64G

# Convert dataset_4 CHGCARs into ChargE3Net's .npy + _atoms.pkl format.
# CPU-only, fast queue. Override INPUT/OUTPUT/LIMIT via --export if needed.

module load anaconda3/2025.12
conda activate ${HOME}/.conda/envs/charge3net

cd /scratch/gpfs/ROSENGROUP/${USER}/charge3net-benchmark

mkdir -p inference_benchmark/logs

INPUT=${INPUT:-/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/dataset_4/label/label}
OUTPUT=${OUTPUT:-/scratch/gpfs/ROSENGROUP/${USER}/charge3net-benchmark/data_preprocessed}
LIMIT=${LIMIT:-1000}
WORKERS=${WORKERS:-16}

echo "Input:   ${INPUT}"
echo "Output:  ${OUTPUT}"
echo "Limit:   ${LIMIT}"
echo "Workers: ${WORKERS}"

srun python inference_benchmark/preprocess_chgcars.py \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --limit "${LIMIT}" \
    --workers "${WORKERS}"
