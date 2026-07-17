# Multi-node Lambda training runbook (`scripts/lambda/MULTINODE.md`)

Drop-in 2-node companion to the single-node runbook (`README.md`). Same data
layout, same configs, same tmux session structure -- just two Lambda
instances coordinating over Ethernet via `torchrun` + NCCL.

The training entrypoint (`src/electrai/entrypoints/train.py`) already reads
`LOCAL_WORLD_SIZE` / `WORLD_SIZE` from the environment and derives
`num_nodes` for the Lightning Trainer. We don't touch model code; this is
purely a launcher + ops doc.

> **Heads-up.** Each Lambda instance has its own `/lambda/nfs/<uuid>`
> filesystem -- they are NOT cross-mounted. Data and checkpoints live on
> each node's local NVMe / NFS independently. Only the S3 checkpoint backup
> on the head node is shared state.

## Topology

```
                            29500/tcp + NCCL data
        Lambda instance A  <----------------------->  Lambda instance B
        (head, NODE_RANK=0)       100 Gbps eth      (worker, NODE_RANK=1)
        8 x H100 80GB SXM                            8 x H100 80GB SXM
        /lambda/nfs/<A-uuid>                         /lambda/nfs/<B-uuid>
        /home/ubuntu (NVMe)                          /home/ubuntu (NVMe)
```

- **Rendezvous:** torchrun static (`--master_addr` + `--master_port=29500`).
  Simpler than `c10d` for two known-IP nodes and matches the runbook's
  "head node IP" mental model. Swap for `--rdzv_backend=c10d` if/when we
  want elastic worker restarts (not needed today).
- **Global batch size goes from 4 -> 16.** The Lightning `Trainer` sees
  `devices="auto"` (= 8) and `num_nodes=2`. See "LR scaling" below.

---

## Pre-flight checklist (in order)

### 1. Provision a second Lambda instance

Pick the same `gpu_8x_h100` SKU. After it boots, SSH in as `ubuntu@<ip>`.

### 2. One-time env setup on the new node

Same as single-node:

```bash
curl -fsSL https://raw.githubusercontent.com/Quantum-Accelerators/electrai/betsy/gga-gga+u-f32/scripts/lambda/setup.sh | bash
```

Then, on the new node:

```bash
aws configure                     # use the betsy-della creds (read + ckpt write)
echo "export WANDB_API_KEY='...'" >> ~/.bashrc
```

### 3. Sync data to the new node's NVMe (~1-3 h, parallel to step 4 below)

```bash
bash ~/electrai/scripts/lambda/data_sync.sh
bash ~/electrai/scripts/lambda/prep_data.sh
```

Each node needs its own full copy of the data on its own NVMe. There is
no cross-node sharing.

### 4. (Head node, in parallel) stage a known-good checkpoint on the head

If you're switching mid-run from the existing 4 GPU run, first cleanly stop
training so `last.ckpt` is consistent:

```bash
# inside the existing electrai-train tmux session on the head node:
tmux send-keys -t electrai-train:train C-c   # graceful stop
# wait for Lightning to checkpoint + exit; verify last.ckpt timestamp is fresh
ls -lh $CKPT_ROOT/last.ckpt
```

Then sync the checkpoint to the worker so any rank can resume:

```bash
# head node -> worker node (run on head):
rsync -avz $CKPT_ROOT/last.ckpt ubuntu@<worker-ip>:$CKPT_ROOT/
```

(Lightning only restores from `last.ckpt` on the global-rank-0 process, so
strictly the worker doesn't need it -- but having both copies makes
disaster recovery cheaper.)

### 5. Find each node's PRIVATE / internal IP

We need the IP that the OTHER node actually routes packets to -- usually a
private 10.x or 172.x address on the 100 Gbps NIC, NOT the public NAT'd
address Lambda lists in their console.

On each node:

```bash
# show all interfaces and their addresses
ip -brief addr

# the IP you want is on the high-bandwidth NIC (look for the 100Gb one,
# usually enp* or ens* with a 10.x or 172.x address)
```

Pick the head node's private IP -- call it `HEAD_IP` for the rest of this
doc. Both nodes will use it as `MASTER_ADDR`.

### 6. Network connectivity check

From the **worker** node:

```bash
ping -c 3 $HEAD_IP                    # latency sanity (should be < 1 ms)
nc -zv $HEAD_IP 29500                 # MASTER_PORT reachable?
```

If `nc` fails, check Lambda's per-instance firewall (cloud-init may have
left iptables open by default, but verify with `sudo iptables -L -n`).

### 7. Find the right `NCCL_SOCKET_IFNAME`

The launcher auto-detects this via `ip route get $MASTER_ADDR`. To
double-check manually on the worker:

```bash
ip -o route get $HEAD_IP | awk '{for(i=1;i<=NF;i++)if($i=="dev")print $(i+1)}'
```

That prints the interface name (e.g. `enp25s0`, `ens9`). If it's `lo`,
`docker0`, or a vlan, something's wrong -- pass `NCCL_SOCKET_IFNAME=...`
explicitly to the launcher.

### 8. NCCL allreduce smoke test (strongly recommended)

Validates that NCCL can actually talk across the two nodes -- way faster
to debug here than inside Lightning.

```bash
# On HEAD:
NODE_RANK=0 MASTER_ADDR=$HEAD_IP NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO \
  uv run torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
    --master_addr=$HEAD_IP --master_port=29500 \
    scripts/lambda/nccl_test.py

# On WORKER (within ~60s of starting the head):
NODE_RANK=1 MASTER_ADDR=$HEAD_IP NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO \
  uv run torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
    --master_addr=$HEAD_IP --master_port=29500 \
    scripts/lambda/nccl_test.py
```

Expect 16 lines of the form `[rank N/16 local=M] allreduce got=120.0
expected=120.0 ok=True` (sum of `0..15` is 120).

If it hangs at "initializing process group", see "Diagnostics" below.

---

## Smoke test (200 samples, 2 epochs)

Same config as single-node smoke, but with `WANDB_MODE_OVERRIDE=disabled`
to keep multi-node tests out of wandb.

On both nodes, start within ~60 s of each other:

```bash
# HEAD:
NODE_RANK=0 MASTER_ADDR=$HEAD_IP WANDB_MODE_OVERRIDE=disabled \
  bash ~/electrai/scripts/lambda/run_training_multinode.sh smoke

# WORKER:
NODE_RANK=1 MASTER_ADDR=$HEAD_IP WANDB_MODE_OVERRIDE=disabled \
  bash ~/electrai/scripts/lambda/run_training_multinode.sh smoke
```

Validate:

- `tmux attach -t electrai-train` on each node shows training advancing.
- Head log shows `world_size=16, num_nodes=2` in Lightning's startup banner.
- Loss curves match the 4 GPU smoke (within stochasticity); if they don't,
  something is wrong with DDP gradient sync.
- ~half the steps per epoch vs single-node 8 GPU (since each rank sees the
  same per-rank batch but global throughput is 2x).

---

## Full run

Once smoke passes, on both nodes (within ~60 s of each other):

```bash
# HEAD:
NODE_RANK=0 MASTER_ADDR=$HEAD_IP WANDB_MODE_OVERRIDE=offline \
  bash ~/electrai/scripts/lambda/run_training_multinode.sh full

# WORKER:
NODE_RANK=1 MASTER_ADDR=$HEAD_IP WANDB_MODE_OVERRIDE=offline \
  bash ~/electrai/scripts/lambda/run_training_multinode.sh full
```

The head node also runs the `backup` and `wandb-sync` tmux windows
(workers skip them; only global rank 0 writes checkpoints, and that
process lives on the head).

---

## LR scaling (open question, flag for review)

Going from 4 -> 16 GPUs means global batch size goes from 4 -> 16
(`batch_size=1` per rank, see `config_gga_gga+u_f32.yaml`).

The current config has `lr: 0.001`. Linear-scaling rule says `lr * 4 =
0.004`; square-root rule says `lr * 2 = 0.002`. The conservative starting
point is somewhere between.

**Recommendation: start with `lr = 0.0025` (2.5x), run 1 epoch, compare
val loss against the 4 GPU baseline at the same epoch.** If loss is
clearly higher than the 4 GPU run's epoch 1, drop back to `lr * 2`. If
loss matches or improves, hold there for a few epochs, then optionally
push to `lr * 3` for the rest of the campaign.

To override the LR without editing the committed config:

```bash
# easiest: copy the config and edit lr in the copy
cp src/electrai/configs/MP/config_gga_gga+u_f32.yaml \
   src/electrai/configs/MP/config_gga_gga+u_f32_2node.yaml
sed -i 's/^lr: .*/lr: 0.0025/' \
   src/electrai/configs/MP/config_gga_gga+u_f32_2node.yaml
# then point SRC_CFG at the new file (or add a "full2node" MODE).
```

This is the one piece of config that genuinely needs a 1-epoch
verification before committing to the full 100-epoch campaign. Don't
skip it.

---

## Switching from the in-flight 4 GPU run

1. Verify what's running:
   `tmux ls -t electrai-train` on `lambda2`.
2. Stop training gracefully so `last.ckpt` is consistent:
   `tmux send-keys -t electrai-train:train C-c`, then wait for
   `last.ckpt` mtime to update and the training process to exit.
3. (Optional, recommended) Bump the S3 backup once so the worker can
   pull from there if the head dies:
   `aws s3 sync $CKPT_ROOT s3://oa-electrai/checkpoints/lambda/`
4. Provision the second instance, follow steps 2-7 above.
5. Run the smoke test (steps "Smoke test" above) to validate cross-node
   DDP works on this hardware pairing.
6. Launch the full multi-node run.

The same `last.ckpt` resumes -- Lightning's `ckpt_path=` argument in
`train.py` is symmetric across world sizes (DDP just shards the loaded
state across whatever ranks are present).

---

## Diagnostics (common multi-node failures)

| Symptom | Likely cause | Fix |
|---|---|---|
| Hangs at "initializing process group", no NCCL output yet | torchrun rendezvous failed (TCP store). | Check `MASTER_ADDR` is reachable; `nc -zv $MASTER_ADDR 29500` from worker. |
| `NCCL WARN ... connection closed by remote peer` followed by hang | NCCL picked the wrong NIC (e.g. docker0 or a vlan). | Set `NCCL_SOCKET_IFNAME=<iface>` explicitly. Use the iface from `ip route get $MASTER_ADDR`. |
| `NCCL timeout` after some training steps | Stragglers / one rank stuck (e.g. dataloader). | Inspect each rank's stack with `py-spy dump --pid <python-pid>`. Often a single bad zarr file. |
| `world_size` doesn't match `nnodes * nproc_per_node` | One side launched with the wrong `--nnodes`. | Check that both launchers used `NUM_NODES=2`. |
| `RuntimeError: Address already in use` on head | A previous run left a torchrun TCPStore on 29500. | `pkill -f torchrun; sleep 3` then relaunch. Or pick a different `MASTER_PORT`. |
| Workers exit instantly with rank 0 unreachable | Wrong `MASTER_ADDR` (used the public IP instead of private). | Re-run step 5 above to pick the private IP. |
| Loss curves diverge between single-node and multi-node | Effective batch size changed; LR not scaled. | See "LR scaling" above. |
| `NCCL: failed to bind to interface` | Firewall blocking 29500 or NCCL's chosen port range. | `sudo iptables -L -n`; on Lambda's default cloud-init image this should be open, but verify. |
| Only one node prints "initializing process group" | Static rendezvous timed out (>~5 min) waiting for the late joiner. | Restart both within ~60 s of each other. |

---

## Tunables (env vars)

Inherited from `run_training.sh`:

| var | default | what |
|---|---|---|
| `REPO_DIR` | `~/electrai` | repo location |
| `DATA_ROOT` | `$NFS_ROOT/data` | local data root |
| `CKPT_ROOT` | `$NFS_ROOT/checkpoints` | local checkpoint dir |
| `S3_CKPT_BUCKET` | `oa-electrai` | S3 bucket for ckpt backups |
| `S3_CKPT_PREFIX` | `checkpoints/lambda-multinode` | ckpt backup prefix |
| `CKPT_BACKUP_S` | `600` | seconds between backups |
| `TMUX_SESSION` | `electrai-train` | tmux session name |
| `WANDB_MODE_OVERRIDE` | (unset) | override `wandb_mode` (`offline` / `disabled`) |

Multi-node specific:

| var | default | what |
|---|---|---|
| `NODE_RANK` | (required) | 0 for head, 1+ for workers |
| `MASTER_ADDR` | (required) | head node's private IP, reachable from all workers |
| `NUM_NODES` | `2` | total node count |
| `NPROC_PER_NODE` | `8` | GPUs per node |
| `MASTER_PORT` | `29500` | torchrun rendezvous port |
| `NCCL_SOCKET_IFNAME` | (auto) | NIC for NCCL; autodetect via `ip route get $MASTER_ADDR` |
