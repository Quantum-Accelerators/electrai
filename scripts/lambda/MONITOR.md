# Monitoring an in-flight Lambda training run

A 100-epoch run is ~26 days. Babysitting that with a human-driven status check
is impractical, so this doc describes the lightweight monitor we've been using:
an hourly LLM-driven check using a fixed prompt + a one-shot status script.

The companion script `monitor_status.sh` produces a single snapshot you can
read directly; the LLM prompt fires the same snapshot hourly and decides
whether anything needs attention.

## What "healthy" looks like (the 3-part liveness rule)

Training is healthy iff **all three** are true:

| Signal | Threshold | What it confirms |
|---|---|---|
| (a) `train.log` mtime | within 60s of now | log file actively being written |
| (b) All GPUs (4 on lambda2, 8 on multi-node) | ≥ 80% util | trainer actively computing |
| (c) `last.ckpt` mtime | within 8h of now | per-epoch checkpoint advance |

**Only flag a stall if at least 2 of the 3 fail.** Single-signal failures are
almost always false alarms.

## Why each non-obvious signal exists

- **Don't trust the visible step counter in `tail train.log`.** The log is
  multi-GB and grows by ~1 MB/min. Each progress update is a `\r`-overwritten
  line, but each overwrite also appends a fresh log entry. The "last 200 lines"
  from a `tail` is therefore a snapshot of ~1 minute of training and looks
  identical between hourly checks even when training is fine. The mtime of the
  file itself is the trustworthy freshness signal.
- **GPU util is the most direct signal training is doing real work.** All four
  H100s should be 90-100% busy. A dataloader stall drops them; a dead trainer
  drops them; a NCCL hang drops them.
- **`last.ckpt` mtime is the per-epoch advance marker.** Lightning writes it
  at the end of each validation, so it cleanly tells you "we finished epoch
  N". With 6-7h epochs, 8h is the right threshold.

## False alarms we've already learned

- `tail` on a fast-writing log returns a snapshot that looks stale between
  consecutive hourly checks. Use mtime + GPU util instead.
- `stat <full_path>` can return ENOENT on NFS/virtiofs while `ls <dir>` of
  the same path works. Don't conclude "file missing" from one `stat`.
- "Backup log entry stuck" can just mean the 10-min cadence hasn't fired yet.
  Check against the loop's cadence, not against arbitrary timestamps.

## Setup: the hourly cron monitor

Inside a Claude Code session, the recurring task is scheduled via `CronCreate`
with this prompt (verbatim, hourly at minute `:27` to avoid herd timing):

```text
check the Lambda H100:4 full training run for config_gga_gga+u_f32 on lambda2.

Run on lambda2 via `ssh lambda2`:
1. `tmux ls 2>&1 | head -3` — verify `electrai-train` session alive
   (expect 3 windows: train, backup, wandb-sync)
2. `date -Iseconds; stat -c '%y %n' \
     /lambda/nfs/<NFS>/checkpoints/train.log \
     /lambda/nfs/<NFS>/checkpoints/<RUN>/last.ckpt 2>&1`
   — liveness via mtimes (NOT step counter)
3. `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader`
   — all should be 90-100%
4. `tmux capture-pane -t electrai-train:train -p -S -100 2>&1 \
     | tr '\r' '\n' | grep -oE 'Epoch [0-9]+: *[0-9]+%.*it/s.*' | tail -2`
   — current epoch/step/it/s from live tmux pane
5. `ls -la /lambda/nfs/<NFS>/checkpoints/<RUN>/ 2>&1 | tail -6`
   — checkpoint files
6. `tail -3 /lambda/nfs/<NFS>/checkpoints/backup.log`
   — S3 ckpt backup loop (10-min cadence)
7. `tail -8 /lambda/nfs/<NFS>/checkpoints/wandb-sync.log`
   — wandb-sync loop; expect periodic "Syncing: ... done." lines

Liveness rule (avoid false alarms): training is healthy iff
(a) train.log mtime within 60s of now AND
(b) all GPUs >= 80% util AND
(c) last.ckpt mtime within 8h of now.
The step counter in the tail of train.log is NOT a reliable freshness
signal. Only flag a stall if at least 2 of those three signals show
>threshold staleness.

Report concisely:
- Current epoch + step + it/s (from tmux pane)
- Latest val_loss_epoch (and delta vs previous)
- New checkpoint files since last check
- wandb-sync state: time of last "Syncing: ... done." vs now
- Any errors / restarts (training exits != 0; wandb sync failures)
- ETA estimate: epochs remaining × current epoch wall-clock

If the strict liveness rule (a&b&c) fails OR a checkpoint hasn't
advanced for >8h OR wandb-sync has logged failures for >2 consecutive
cycles, surface the failure and consider intervention.
```

For multi-node (see [PORT_PLAN.md](PORT_PLAN.md) Phase 7), add a second SSH
target for the worker node and check its GPUs + tmux separately.

## Ad-hoc usage

```sh
bash scripts/lambda/monitor_status.sh                    # default: lambda2, full_gga_gga+u_f32
bash scripts/lambda/monitor_status.sh lambda3-a          # different host
bash scripts/lambda/monitor_status.sh lambda3-a smoke_gga_gga+u_f32_smoke
```

## Intervention thresholds

| Symptom | Action |
|---|---|
| Liveness rule (a+b+c) fails for 1 cycle | wait a cycle — likely transient |
| Liveness rule fails for 2+ cycles | ssh in, check `ps aux | grep python`, GPU XID errors in `dmesg` |
| `last.ckpt` >8h old | ssh in, check if validation is hung; consider killing + resuming from S3 mirror |
| wandb-sync fails for 2+ cycles | check `wandb-sync.log`; if backend is down, sync will catch up later — don't intervene unless training itself is also failing |
| Worker node unreachable (multi-node only) | DDP will time out and trainer will auto-resume; verify worker GPU/tmux when reachable again |
| `nvidia-smi` reports XID errors | hardware issue; preserve state via S3, file Lambda support ticket, consider switching instances |

## What this monitor doesn't do

- It doesn't auto-restart anything. The `while true` resume loop in
  `run_training.sh` handles trainer crashes; the wandb-sync and backup
  windows have their own loops too. The monitor's job is to surface
  failures the operator should investigate.
- It doesn't watch loss curves for divergence — that's wandb's job. The
  monitor checks that wandb is *receiving* data, not that the data is
  good.
- It doesn't track Lambda billing. Cost is roughly $11.96/hr (4×) or
  $47.84/hr (16×); back-of-envelope from "epochs left × wall-clock per
  epoch × $/hr".
