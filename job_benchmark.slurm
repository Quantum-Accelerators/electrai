#!/bin/bash
#SBATCH --job-name=electrai-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/ROSENGROUP/${USER}/logs/benchmark-%j.out

module purge
module load anaconda3/2025.6
conda activate electrai
module load proxy/default

cd /scratch/gpfs/ROSENGROUP/${USER}/electrai2

export PYTHONPATH=$(pwd)
export PYTORCH_ALLOC_CONF=expandable_segments:True

SCRATCH_DIR=/scratch/gpfs/ROSENGROUP/${USER}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE=${SCRATCH_DIR}/profiles/benchmark_results.json
TIMESTAMPED_FILE=${SCRATCH_DIR}/profiles/benchmark_results_${TIMESTAMP}.json
REPORT_FILE=${SCRATCH_DIR}/profiles/compute_estimates.md

mkdir -p ${SCRATCH_DIR}/profiles

# Phase 1-3: Run benchmark (single GPU, no DDP)
uv run --no-sync python src/electrai/scripts/benchmark.py \
    --config src/electrai/configs/MP/config_benchmark.yaml

# Move results to $SCRATCH and generate report
if [ -f benchmark_results.json ]; then
    mv benchmark_results.json ${RESULTS_FILE}
    cp ${RESULTS_FILE} ${TIMESTAMPED_FILE}
    echo "Results saved to ${TIMESTAMPED_FILE}"

    # Generate report
    uv run --no-sync python src/electrai/scripts/analyze_benchmark.py \
        --results ${RESULTS_FILE} \
        --output ${REPORT_FILE}

    echo "Report generated at ${REPORT_FILE}"
else
    echo "ERROR: ${RESULTS_FILE} not found"
    exit 1
fi
