#!/bin/bash
#SBATCH --job-name=w128-test-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/ROSENGROUP/bb9080/logs/w128-test-eval-%j.out

# Evaluate the W128 lr2e-3 checkpoint (epoch 55, val 0.005785) on the withheld
# GGA + GGA+U test set (split_capped.json test keys: 1700 + 524 samples).
# Config: src/electrai/configs/MP/config_gga_gga+u_w128_test_della.yaml

module purge
module load anaconda3/2025.6
conda activate electrai
module load proxy/default
export PATH=$HOME/.local/bin:$PATH
# Variable grid sizes fragment the caching allocator; expandable segments
# lets large per-sample activations reuse freed reserved memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch/gpfs/ROSENGROUP/bb9080/electrai

EVAL=/scratch/gpfs/ROSENGROUP/bb9080/w128_test_eval
CKPT_SRC=/scratch/gpfs/ROSENGROUP/bb9080/checkpoints/w128_ckpt_epoch55_val0.005785.ckpt

mkdir -p ${EVAL}/ckpt ${EVAL}/results
# test entrypoint expects <ckpt_path>/last.ckpt
ln -sfn ${CKPT_SRC} ${EVAL}/ckpt/last.ckpt

echo "=== W128 test-set eval ==="
echo "Checkpoint: ${CKPT_SRC}"
echo "Results:    ${EVAL}/results"
echo "Start:      $(date)"
echo ""

srun uv run --no-sync python ./src/electrai/entrypoints/main.py test \
    --config src/electrai/configs/MP/config_gga_gga+u_w128_test_della.yaml

echo ""
echo "End: $(date)"

METRICS=${EVAL}/results/metrics.csv
if [ ! -f ${METRICS} ]; then
    echo "ERROR: metrics.csv not found at ${METRICS}"
    exit 1
fi
N_ROWS=$(tail -n +2 ${METRICS} | wc -l)
echo "metrics.csv rows: ${N_ROWS} (expected 2224)"
