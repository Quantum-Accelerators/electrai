# EC2 resident monitor — active operator for the Lambda run

This is the **active** counterpart to [MONITOR.md](MONITOR.md). MONITOR.md
describes an *observe-only* hourly check (it explicitly "doesn't auto-restart
anything"). This doc describes a dedicated always-on EC2 box running a resident
Claude agent that uses the same liveness rule and snapshot script, but **also
attempts a bounded set of safe fixes on its own and escalates to Slack only when
a fix fails or the situation is outside its allowed actions.**

Use one or the other, not both pointed at the same run.

## Why a separate box
The MONITOR.md cron lives *inside* a Claude Code session — it needs a live
session running somewhere (a laptop). A 100-epoch run is ~26 days, so the
watcher must be durable and independent of both your laptop and the training
box. A tiny EC2 instance gives you: 24/7 uptime, co-location with
`s3://oa-electrai` (us-east-1), an IAM instance role instead of static keys, and
a blast radius separate from the GPUs it babysits.

## Components (all in `scripts/lambda/`)
| File | Role |
|---|---|
| `monitor_status.sh` | one-shot read-only snapshot of one node (existing) |
| `monitor_status_all.sh` | wraps the above over every node in `$HOSTS` (head + worker) |
| `monitor_agent_prompt.md` | the operator prompt fed to Claude each tick |
| `monitor_loop.sh` | resident supervisor loop (run by systemd) |
| `monitor_settings.json` | Claude permission allowlist (fail-closed) |
| `monitor.env.example` | EnvironmentFile template (targets + secrets) |
| `monitor_watchdog.sh` | independent "who watches the watcher" check |
| `systemd/electrai-monitor.service` | runs the loop, `Restart=always` |
| `systemd/electrai-monitor-watchdog.{service,timer}` | runs the watchdog every 30 min |

### How a tick works
1. The loop prepends a live **CONTEXT** header (HOSTS, RUN, journal path,
   maintenance flag, snapshot command) to `monitor_agent_prompt.md`.
2. Claude reads the tail of the **journal** (its memory across ticks), runs
   `monitor_status_all.sh`, judges health by MONITOR.md's 3-part liveness rule,
   and either does nothing, remediates, or escalates.
3. Claude's reply is appended to the journal. Continuity is the journal, **not**
   the conversation buffer — that's what makes this survivable over 26 days and
   across restarts/token refreshes.
4. A successful tick touches the **heartbeat** file; the watchdog keys off that,
   so a silently failing Claude (expired token, quota, network) still gets
   caught even though the journal header updates every tick.

## REMEDIATION RUNBOOK (the agent reads this section)

The training script already self-heals transient crashes (`run_training.sh` runs
the trainer in a `while true; … sleep 30` loop and Lightning auto-resumes from
`last.ckpt`). So the agent's autonomous scope is deliberately narrow: the cases
the self-heal loop **can't** cover. Confirm a real stall with the liveness rule
(>= 2 of 3 failing) before acting.

### Allowed autonomous actions (safe, idempotent, reversible)
| Situation | Action |
|---|---|
| `electrai-train` tmux session absent entirely | relaunch: `ssh <head> 'cd ~/electrai && bash scripts/lambda/run_training.sh full'` (auto-resumes) |
| A loop window died (backup or wandb-sync window gone, training fine) | restart just that window in the existing session |
| Train alive but wedged >= 2 cycles (GPUs idle + `train.log` stale → NCCL hang) | kill + relaunch the train window once (`NCCL_ASYNC_ERROR_HANDLING=1` should make resume clean) |
| (multi-node) worker tmux died but host reachable | restart the worker training command |
| Single-cycle / single-signal failure | wait one tick — likely transient (per MONITOR.md) |

### Circuit breaker
At most **2** autonomous restart actions for the same issue within a rolling
**60-minute** window — count them from your journal recall. On the 3rd
occurrence, stop remediating and **escalate** instead of thrashing.

### Escalate-only → Slack (never autonomous)
| Situation | Why escalate |
|---|---|
| `nvidia-smi` / `dmesg` XID hardware errors | hardware; preserve state, human picks whether to switch instances (cost) |
| Worker host *unreachable* / instance down (not just tmux) | provisioning is a cost decision for a human |
| `last.ckpt` still > 8h old after one allowed restart | validation genuinely hung; needs a look |
| Anything risking checkpoint/data loss; disk full | irreversible |
| Persistent failure past the circuit breaker | stop and page |
| Its own auth failures (Claude token / wandb / AWS / SSH) | the agent can't fix itself — surfaced by the watchdog |
| wandb-sync failing while training is healthy | **log only, do not escalate** (per MONITOR.md) |

When unsure of the root cause, do not act on the cluster — escalate.

### Maintenance mode
During the H100:4 → H100:16 cutover (see [PORT_PLAN.md](PORT_PLAN.md)) there is a
lot of expected churn. Pause autonomous remediation:
```sh
touch  ~/.config/electrai-monitor/MAINTENANCE   # observe + journal only
rm     ~/.config/electrai-monitor/MAINTENANCE   # resume normal operation
```

## Provisioning the box

### 1. Instance
- `t4g.medium` (2 vCPU / 4 GB, ARM) — no GPU; Claude runs on Anthropic's
  servers, the box only SSHes/parses/loops. ~$24/mo.
- Ubuntu 24.04 LTS, **us-east-1** (co-located with `s3://oa-electrai`).
- 30 GB gp3 EBS. Allocate an **Elastic IP** (stable address for your SSH in and
  for allowlisting on the Lambda boxes).
- Security group: inbound SSH (22) **from your IP only**; outbound all.

### 2. IAM instance role (no static keys)
Attach a role with **read-only** access to the checkpoint prefix — restores
happen on the Lambda box with its own creds, so the monitor never needs write:
```
s3:GetObject, s3:ListBucket on arn:aws:s3:::oa-electrai (prefix checkpoints/lambda/*)
```

### 3. Software
```sh
sudo apt-get update && sudo apt-get install -y git tmux awscli jq curl
curl -LsSf https://astral.sh/uv/install.sh | sh           # optional, parity with Lambda
# Node LTS + Claude Code (per current install docs), then:
claude setup-token        # interactive once — binds your Max plan (long-lived token in ~/.claude)
git clone -b betsy/gga-gga+u-f32 git@github.com:Quantum-Accelerators/electrai.git ~/electrai
```

### 4. SSH to the Lambda boxes (dedicated, revocable key)
Do **not** copy your personal `id_ed25519_lambda`. Mint a key just for the
monitor and authorize it on each Lambda node:
```sh
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_monitor -N ''
# add ~/.ssh/id_ed25519_monitor.pub to ~/.ssh/authorized_keys on lambda2 (and lambda3-a/-b)
```
Replicate the host aliases in `~/.ssh/config` (`lambda2`, `lambda3-a`,
`lambda3-b`) pointing at `id_ed25519_monitor`, then verify:
```sh
ssh lambda2 nvidia-smi
bash ~/electrai/scripts/lambda/monitor_status.sh lambda2
```
Consider locking the Lambda nodes' SSH inbound to the EC2 Elastic IP + your laptop.

### 5. Config + secrets
```sh
mkdir -p ~/.config/electrai-monitor ~/.local/state/electrai-monitor
cp ~/electrai/scripts/lambda/monitor.env.example ~/.config/electrai-monitor/monitor.env
chmod 600 ~/.config/electrai-monitor/monitor.env
# edit: HOSTS, RUN, SLACK_WEBHOOK_URL, WANDB_API_KEY
```

### 6. Install the services
```sh
sudo cp ~/electrai/scripts/lambda/systemd/electrai-monitor*.service /etc/systemd/system/
sudo cp ~/electrai/scripts/lambda/systemd/electrai-monitor-watchdog.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now electrai-monitor.service
sudo systemctl enable --now electrai-monitor-watchdog.timer
journalctl -u electrai-monitor -f          # or: tail -f ~/.local/state/electrai-monitor/journal.md
```

### 7. Smoke test (before trusting it)
1. Confirm a healthy tick lands a `STATUS: HEALTHY` block in the journal.
2. At a safe moment, `ssh lambda2 'tmux kill-session -t electrai-train'` and
   confirm the agent detects the stall, relaunches the run, and logs `STATUS:
   ACTED`. Watch that the trainer resumes from `last.ckpt`.
3. Force an escalate-only case (e.g. set `HOSTS` to an unreachable host) and
   confirm a Slack message arrives.
4. Stop `electrai-monitor.service` and confirm the watchdog Slacks within its
   interval.

## Cost
~$24/mo instance + ~$3/mo EBS + negligible egress. Claude runs on your Max
subscription token, so no per-token API bill — but a continuous loop consumes
Max usage; tune `MONITOR_INTERVAL` (default 15 min) and `MONITOR_MODEL` (default
Sonnet) if you bump limits. Trivial next to the ~$11.96/hr (4×) / ~$47.84/hr
(16×) training spend.

## Multi-node cutover checklist
1. `touch ~/.config/electrai-monitor/MAINTENANCE` before the cutover.
2. Edit `monitor.env`: `HOSTS="lambda3-a lambda3-b"`, `RUN="full_gga_gga+u_f32_16x"`.
3. Add `lambda3-a` / `lambda3-b` to `~/.ssh/config` and authorize the monitor key.
4. (Already in `monitor_settings.json`: ssh allow entries for both.)
5. `sudo systemctl restart electrai-monitor` and confirm a healthy multi-node tick.
6. `rm ~/.config/electrai-monitor/MAINTENANCE` to re-enable remediation.

## Open items before go-live
- **Sign-off on the allowed-action list above** — this is the one place the
  monitor touches a live, expensive run.
- Confirm the Max-plan token survives unattended (the watchdog catches expiry,
  but plan for periodic `claude setup-token` refresh).
