# Plan: port the in-flight H100:4 run to H100:16 (2 nodes × 8 GPUs)

Companion to [MULTINODE.md](MULTINODE.md) — that file is the static operator
runbook; this file is the migration plan for taking a *running* 4× campaign
and porting it onto a 2-node 16-GPU cluster with the lowest risk and the
earliest abort signal.

Status at the time of writing: H100:4 on `lambda2`, ~48h in, ~epoch 7,
`ckpt_epoch=06_val_loss=0.011800.ckpt` saved. Multi-node scaffolding lives
in `scripts/lambda/{run_training_multinode.sh, MULTINODE.md, nccl_test.py}`.

## Why this is risky enough to need a plan

Going from 1 node to 2 changes several things at once:

- **DDP world size 4 → 16**, so the effective batch size goes 4 → 16 (bs=1
  per rank). Need to scale `lr`.
- **All-reduce hops over Ethernet**, not NVLink. Cheap for our 12M-param
  model in theory, but NCCL config is hardware-specific.
- **Doubled failure surface** — two instances means ~2× the probability of
  hardware/network failure stopping the run.
- **Capacity uncertain** — Lambda has been showing 8× H100 SXM unavailable
  on demand. Provisioning two simultaneously may take time.

Goal: don't commit to 16× until we have measured proof it's worth it.

## Decision gates

> **Smoke gate:** the 16× smoke must hit **≥3.5× the 4× run's
> samples/sec** (≥15.4 samples/sec ≈ ≥3.85 it/s at world-size 16). Below
> 2.5× → abort.
>
> **LR gate:** one 16× epoch resumed from `last.ckpt` with a candidate `lr`
> must give a `val_loss` no worse than 1.5× the 4× run's value at the same
> epoch index. If it spikes or plateaus, try a different `lr` (cheap; one
> epoch each).

Hitting both gates before cutover bounds the downside of a bad migration.

## Phase 1 — Provision (~30 min)

1. Spin up two Lambda 8× H100 SXM instances. Call them `lambda3-a` and
   `lambda3-b`. Configure `~/.ssh/config` aliases (Identity, ProxyJump if
   needed, ForwardAgent yes).
2. On each: clone the repo, check out `betsy/gga-gga+u-f32`, run
   `scripts/lambda/setup.sh`.
3. On each: `aws configure` (region `us-east-1`) and append
   `WANDB_API_KEY` to `~/.bashrc`.
4. Verify each has a `/lambda/nfs/...` mount and ≥1.4 TB free NVMe.

## Phase 2 — Data on both nodes (~3–4h, parallel)

Each node needs its own NVMe copy — NFS is too slow (we measured 4× drop in
throughput on the current run). In two parallel terminals:

```sh
ssh -A lambda3-a 'cd ~/electrai && tmux new-session -d -s data-sync \
  "bash scripts/lambda/data_sync.sh 2>&1 | tee -a ~/data_sync.log"'
ssh -A lambda3-b 'cd ~/electrai && tmux new-session -d -s data-sync \
  "bash scripts/lambda/data_sync.sh 2>&1 | tee -a ~/data_sync.log"'
```

When each completes (`OK.` line in `~/data_sync.log`), run on **both**:

```sh
DATA_ROOT=~/data bash ~/electrai/scripts/lambda/prep_data.sh
```

Do the Phase 3 network prep in parallel while the data syncs.

## Phase 3 — Network & NCCL validation (~30 min)

1. **Find the head node's reachable IP** (private/internal, not the public
   NAT'd address):

   ```sh
   ssh lambda3-a 'ip -4 addr show | grep -A1 -E "(ens|enp|eth)" | grep inet'
   ```

   Pick the 10.x or 172.x address. Set `MASTER_ADDR`.

2. **Confirm worker → head reach on port 29500**:

   ```sh
   ssh lambda3-a 'nc -l 29500' &
   ssh lambda3-b "nc -zv $MASTER_ADDR 29500"
   ```

   Should succeed. If blocked, open the port or pick another and set
   `MASTER_PORT`.

3. **Find the right NCCL socket interface** (from the worker):

   ```sh
   ssh lambda3-b "ip route get $MASTER_ADDR" \
     | awk '{for (i=1;i<=NF;i++) if ($i=="dev") print $(i+1)}'
   ```

   Set `NCCL_SOCKET_IFNAME` to that interface name.

4. **Run the NCCL smoke** — `scripts/lambda/nccl_test.py` does a
   cross-node all-reduce. Expect <30s success.

   If it hangs or NCCL errors out, **fix here, not in real training** —
   debugging NCCL while losing real training time is much harder. Common
   fixes: wrong `NCCL_SOCKET_IFNAME`, wrong IP family, firewall on the
   NCCL ephemeral port range.

## Phase 4 — Multi-node smoke (~20 min)

Goal: validate the loader + DDP topology + throughput **before** committing.

```sh
# on head, separate ssh session per node
ssh -A lambda3-a 'cd ~/electrai && DATA_ROOT=~/data NUM_NODES=2 NODE_RANK=0 \
  MASTER_ADDR=10.x.x.x WANDB_MODE_OVERRIDE=disabled \
  bash scripts/lambda/run_training_multinode.sh smoke'

# on worker, same time
ssh -A lambda3-b 'cd ~/electrai && DATA_ROOT=~/data NUM_NODES=2 NODE_RANK=1 \
  MASTER_ADDR=10.x.x.x WANDB_MODE_OVERRIDE=disabled \
  bash scripts/lambda/run_training_multinode.sh smoke'
```

What to measure:

- World size = 16 in the Lightning startup log (`Initializing distributed:
  GLOBAL_RANK: 0, MEMBER: 1/16`)
- Per-step `it/s` from the live tmux pane on the head node
- `val_loss` after 2 epochs (expect ~0.10, comparable to the 4× smoke)

**Decision point #1 — smoke gate.** If samples/sec ≥ 15.4 (i.e.
it/s × 16 ≥ 15.4), continue. If not, abort and stay on 4×.

## Phase 5 — LR-scaling 1-epoch validation (~3h, recommended)

Going 4 → 16 ranks quadruples the effective batch. Standard scaling is
either √4 = 2× `lr` or linear 4× `lr`. The config is at `lr: 0.001`.

1. **Copy `last.ckpt` from the 4× run to the head node's NFS:**

   ```sh
   ssh lambda3-a 'mkdir -p /lambda/nfs/<head-nfs>/checkpoints/full_gga_gga+u_f32 \
     && aws s3 cp s3://oa-electrai/checkpoints/lambda/full_gga_gga+u_f32/last.ckpt \
     /lambda/nfs/<head-nfs>/checkpoints/full_gga_gga+u_f32/last.ckpt'
   ```

2. **Make a candidate config** at `configs/MP/config_gga_gga+u_f32_16x.yaml`
   based on the full config, with:
   - `lr: 0.0025` (2.5×, the geometric midpoint of √ and linear)
   - `run_name: gga_gga+u_f32_16x` (fresh wandb run)
3. **Run 1 epoch** from that ckpt using the new config across both nodes.
4. **Compare** the resulting `val_loss` to the 4× trajectory at the same
   epoch index.

**Decision point #2 — LR gate.** If `val_loss` is monotonically descending
and within 50% of the 4× value at the same epoch, commit. If it spikes or
plateaus higher, try `lr: 0.004` (4×, linear) or `lr: 0.0015` (1.5×) — each
is a cheap 1-epoch experiment.

## Phase 6 — Cutover (~10 min)

1. **Stop the 4× run cleanly.** Two options:
   - Wait for current epoch to end. Worst-case ~6h cost; lossless.
     Recommended.
   - Hard-stop now (loses progress within the current epoch; Lightning
     resumes from the last completed-epoch ckpt anyway).

   ```sh
   ssh lambda2 'tmux kill-session -t electrai-train'
   ```

2. Wait for the lambda2 wandb-sync window to flush the final state, then
   `tmux kill-session -t electrai-train` on lambda2 if not already.
3. **Launch real 16× training** with the chosen `lr`:

   ```sh
   # head
   ssh -A lambda3-a 'cd ~/electrai && DATA_ROOT=~/data NUM_NODES=2 NODE_RANK=0 \
     MASTER_ADDR=10.x.x.x WANDB_MODE_OVERRIDE=offline \
     bash scripts/lambda/run_training_multinode.sh full'
   # worker
   ssh -A lambda3-b 'cd ~/electrai && DATA_ROOT=~/data NUM_NODES=2 NODE_RANK=1 \
     MASTER_ADDR=10.x.x.x WANDB_MODE_OVERRIDE=offline \
     bash scripts/lambda/run_training_multinode.sh full'
   ```

4. Confirm the head node has 3 tmux windows: train, backup, wandb-sync.
   Worker has 1: train.
5. Tear down lambda2 — but leave it up for 30 min as a fallback in case
   the 16× run shows an early problem.

## Phase 7 — Monitoring the 16× run

Most of the existing hourly cron monitor transfers with minor edits:

| Aspect | H100:4 (lambda2) | H100:16 (lambda3-a head + lambda3-b worker) |
|---|---|---|
| ssh target | `lambda2` | `lambda3-a` (head; canonical state lives here) |
| GPU check | 4 GPUs ≥ 80% | 8 GPUs on each node ≥ 80% |
| Per-step check | `tmux capture-pane -t electrai-train:train` | same, on head |
| Checkpoints | local NFS + S3 | head's NFS + S3 (worker writes nothing) |
| wandb-sync | on lambda2 | on head (lambda3-a) |
| **Extra check** | n/a | also `ssh lambda3-b` to confirm worker tmux + GPUs alive |

### New failure modes

- **Worker rank disappears** → DDP times out → head ranks exit → auto-resume
  starts back from `last.ckpt`. Cron should ssh the worker every cycle and
  confirm its training tmux + nvidia-smi are alive. Wasted time if the
  worker is dead a long time.
- **NCCL silent hang** — training stops moving but no error. Catch via the
  existing liveness rule (train.log mtime stale + GPUs idle). With
  `NCCL_ASYNC_ERROR_HANDLING=1` already set in the script, this should
  surface as a clean exception → auto-resume.
- **wandb run id changes** from `8yzlii32` to whatever the 16× run's
  `offline-run-*` directory becomes. Update the wandb-sync log path + cron
  monitor to the new offline dir on the head.

## Concerns / risks (ranked by likelihood × impact)

1. **HIGH — Multi-node networking misconfig.**
   `NCCL_SOCKET_IFNAME` and the routable IP require real-hardware
   verification. **Do not skip the Phase 3 NCCL smoke.** Mitigation:
   `NCCL_DEBUG=INFO` on first run, ready to revisit.
2. **HIGH — Throughput doesn't scale enough.**
   <2.5× of current and we just paid setup time + a few hours of double
   compute for nothing. The decision gates above catch this in <8h
   instead of mid-campaign.
3. **MED — LR scaling lands wrong on first try.**
   Multi-epoch validation is too expensive; one-epoch is what we can
   afford. If the model diverges, worst case is one bad epoch (~$300)
   before we revert.
4. **MED — Doubled instance failure surface.**
   Two H100:8 instances means ~2× the probability of one dying mid-run.
   S3 checkpoint backup already mitigates total data loss; auto-resume
   handles single failures. We lose at most the in-progress epoch.
5. **MED — Lambda capacity for the 2nd instance.**
   Single-node H100:8 has been intermittently unavailable.
   File a Lambda support ticket in parallel with this plan to gauge
   timing.
6. **LOW — wandb dashboard discontinuity.**
   New run id; the loss curve will look like a fresh run starting at the
   ckpt-6 loss value. Paper over with wandb run groups, or just accept it.
7. **LOW — NFS visible only on head.**
   Worker doesn't need NFS (only rank-0 writes). Already handled in the
   multi-node script.
8. **LOW — Cost overrun in setup phase.**
   Even with everything going wrong, 24h of double-instance dead time
   ≈ $575. Acceptable.

## When to actually do this

Don't migrate until at least one of these is true:

- Lambda confirms 2× H100:8 instances are available simultaneously
- val_loss curve on H100:4 plateaus enough that we expect to *need* the
  full 100 epochs (right now it's monotonic, dropping ~5% epoch-over-epoch
  — could plateau at any epoch)
- A specific deadline (e.g. April 2026 funding milestone) forces our hand

If we hit any of these, this plan is ~12h end-to-end from "instances
provisioned" to "real training resumed at 16×". The pre-drafted scripts
handle most of the mechanics; the human work is provisioning, NCCL config
tuning, and the LR validation read.
