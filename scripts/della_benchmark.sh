#!/bin/bash
#SBATCH --job-name=electrai-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=benchmark-%j.log

# Match Betsy's Della setup
module load anaconda3/2025.6
conda activate electrai
module load proxy/default

# Use ELECTRAI_DIR if set, otherwise the repo containing this script
ELECTRAI_DIR="${ELECTRAI_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ELECTRAI_DIR"

export PYTHONPATH=$(pwd)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Use dataset_4 with 100 samples for a quick comparison
# (same as Modal benchmark being run in parallel)
#
# To run:
#   sbatch scripts/della_benchmark.sh
#
# Or interactively (if on a GPU node):
#   bash scripts/della_benchmark.sh

NUM_GPUS=${SLURM_GPUS_PER_NODE:-4}
SAMPLES=100
EPOCHS=2
CONFIG=/tmp/della_benchmark_config.yaml

# Write config matching Betsy's setup
cat > "$CONFIG" << 'YAML'
data:
  _target_: electrai.dataloader.dataset.RhoRead
  root: /scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/dataset_4/mp_filelist.txt
  split_file: null
  precision: f32
  batch_size: 1
  train_workers: 8
  val_workers: 2
  pin_memory: false
  val_frac: 0.005
  drop_last: false
  augmentation: false
  random_seed: 42

model:
  _target_: electrai.model.resunet.ResUNet3D
  in_channels: 1
  out_channels: 1
  n_channels: 32
  n_residual_blocks: 1
  kernel_size: 5
  depth: 2
  use_checkpoint: false

precision: 32
epochs: 2
lr: 0.01
weight_decay: 0.0
warmup_length: 1
gradient_clip_value: 5
wandb_mode: online
entity: PrinceOA
wb_pname: elf-net-ci-test
ckpt_path: /tmp/checkpoints
YAML

# Limit to first N samples by writing a truncated filelist
head -n "$SAMPLES" /scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/dataset_4/mp_filelist.txt > /tmp/della_bench_filelist.txt

# Patch config to use truncated filelist
sed -i "s|root:.*|root: /tmp/della_bench_filelist.txt|" "$CONFIG"
# RhoRead uses the filelist's parent dir to find data/label subdirs,
# so we need a symlink
ln -sf /scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/dataset_4/data /tmp/data
ln -sf /scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/dataset_4/label /tmp/label

echo "=== Della Benchmark ==="
echo "GPUs: $NUM_GPUS"
echo "Samples: $SAMPLES"
echo "Epochs: $EPOCHS"
echo "Config: $CONFIG"
echo ""

START=$(date +%s)

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    ./src/electrai/entrypoints/main.py train \
    --config "$CONFIG"

END=$(date +%s)
ELAPSED=$((END - START))

echo ""
echo "=== Della Benchmark Complete ==="
echo "Wallclock: ${ELAPSED}s"
echo "GPUs: $NUM_GPUS"
echo "Samples: $SAMPLES"
echo "Epochs: $EPOCHS"
