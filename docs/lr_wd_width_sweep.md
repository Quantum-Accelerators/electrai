# LR / weight-decay sweep across width (W32 / W64 / W96)

Status: stages A+B COMPLETE (launched 2026-07-30/31, finished 2026-07-31 —
16 + 15 trials, all succeeded; results below). Stage C not launched.

## Results (see W&B `mp-gga-ggau-lrwd`; aggregate with scripts/review_lrwd_sweep.py)

Best val NMAE% on the 8-epoch/12K proxy. Noise bar from the three verbatim
anchor reruns: **±0.04–0.07 points** (1.914→1.968, 1.592→1.665, 1.524→1.566).

| lr | W32 | W64 | W96 |
|---------|-------|-------|-------|
| 2.5e-4 | 2.330 | 1.998 | 1.943 |
| 5e-4 | 2.133 | 1.841 | 1.836 |
| 1e-3 | 1.935 | 1.708 | 1.757 |
| 2e-3 | **1.914** | 1.759 | **1.524** |
| 4e-3 | 1.954 | **1.592** | 1.562 |
| 8e-3 | — | 1.580 | — |

| wd @ lr* (AdamW) | W32 | W64 | W96 |
|---------|-------|-------|-------|
| 0 (+rerun) | 1.914 / 1.968 | 1.592 / 1.665 | 1.524 / 1.566 |
| 1e-4 | 1.968 | 1.631 | 1.535 |
| 1e-3 | 1.995 | 1.610 | 1.640 |
| 1e-2 | 1.970 | 1.654 | 1.549 |
| cross (lr*/2, 1e-3) | 1.904 | 1.638 | 1.731 |

Conclusions:

1. **lr\* does not shrink with width** — every width sits in a flat 2–4e-3
   basin (differences at/inside the noise bar). The production lr=1e-3 is
   suboptimal at all widths, and the penalty grows with width: ~1% relative
   at W32, ~7% at W64, ~13% at W96. Extrapolation to W128/W160: use 2e-3.
2. **Weight decay is neutral across 1e-4–1e-2 at every width** — even on
   this subset, where overfitting bites ~9× earlier than at full scale, so
   at 111K it has even less room to help. Recommend keeping **wd=0**
   (continuity with all incumbents); wd up to 1e-2 is demonstrably safe if
   ever wanted for other reasons. (Only outlier: W96 @ 1e-3 = 1.640,
   marginally past the bar; with 1e-2 neutral on both sides it reads as an
   unlucky draw, not a trend.)
3. W64's stage-A curiosities dissolved: the 8e-3 "win" (1.580 vs 1.592) and
   the 2e-3 dip (1.759; cross term at same lr with wd landed 1.638) are
   both within run-to-run noise.
4. Width ordering at tuned recipes is unchanged: W96 1.524 < W64 1.592 <
   W32 1.914.

**Stage C recipes**: W32 (2e-3, 0), W64 (4e-3, 0), W96 (2e-3, 0) — with
2e-3 defensible everywhere given the flat basin.

## Why

Every rung of the width ladder (`mp-gga-ggau-width`: W64, W96, W128, W160)
trains with the same recipe — `lr: 0.001`, `weight_decay: 0.0` — inherited
from the June W64 run. For Adam in standard parametrization the optimal LR
typically *shrinks* as width grows (roughly ∝ 1/width in the μP limit), so
the wider rungs are plausibly mistuned and the width-scaling conclusions
conflate capacity with recipe mismatch. W96's 17.8% win over W64 survived
that handicap; the question is whether the gaps (and the W128/W160 verdicts)
change once each width gets its own recipe.

This sweep tunes LR and WD independently at widths 32, 64, 96, then uses the
per-width optima to (a) re-examine the width ranking and (b) extrapolate a
recipe for W128/W160 rather than sweeping at those (much more expensive)
widths.

## Trial design (proxy task)

Full production runs are ~5 days; sweep trials compress two axes and change
nothing else:

- **Data**: same staged capped dataset (111,257 ids). Train is subsampled to
  ~12K structures (frac 0.1112 of each functional's capped train split, so
  the 76/24 gga/gga+u mix is preserved: 9,139 + 2,862). **Validation is the
  untouched production val set** (847 + 262 = 1,109), so trial `val_loss` is
  on the same yardstick as the production curves. The subset lives in two
  small split JSONs checked into the repo (`data/MP/sweep_splits/`) and
  referenced by repo-relative path — they ride along in the submit.sh bundle,
  so no bucket upload and no interaction with the `.staged.ok` warm-node
  skip.
- **Schedule**: `epochs: 8`, `warmup_length: 1` — each trial runs its own
  complete warmup+cosine schedule, compressed. Comparing LRs mid-schedule is
  misleading; comparing completed short schedules preserves ranking much
  better. 8 epochs × 12K ≈ 24K optimizer steps ≈ 0.9 production epochs.
- **Hardware/batch**: 1 node, GB200x4, DDP, batch 1/GPU — the *same
  effective batch (4) as production*, so the tuned LR transfers directly.
  (A single-GPU trial would change the effective batch and invalidate it.)
- Everything else identical to `config_gga_gga+u_w96.yaml`: bf16-mixed, no
  activation checkpointing, Adam betas (0.9, 0.99), offline W&B + sidecar.

Per-trial wall-clock (from measured production step times): W96 ≈ 4 h,
W64 ≈ 3 h, W32 ≈ 2–2.5 h.

## Stage A — LR grid at wd = 0

`lr ∈ {2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3}` × `width ∈ {32, 64, 96}` = **15
trials, ≈ 50 node-hours**. Factor-2 spacing matches the flatness of Adam LR
basins; the 16× range is centered on the incumbent 1e-3 and wide enough for
a ~3× shift across 32→96.

Decision rule per width: `lr*` = argmin of final-epoch `val_loss` (sanity:
best-epoch val and curve shape agree; a diverged/NaN trial is a top-edge
signal, likely at 4e-3 for the wider models). **If a width's optimum lands
on a grid edge, extend one more factor-2 point before moving on.** The old
Optuna tier-1 result (lr ≈ 3.5e-3 best for W32, different data/precision)
hints W32 may press the top edge.

## Stage B — WD grid at lr*

Weight decay uses **AdamW** (`optimizer: adamw`, added to
`lightning.py`) — the incumbent plain Adam applies `weight_decay` as
coupled L2, which gets rescaled per-parameter by the adaptive denominator
and makes values neither interpretable nor comparable across widths. At
wd = 0 the two optimizers are identical, so nothing about existing runs or
Stage A changes.

Per width, 5 trials:

- `wd ∈ {1e-4, 1e-3, 1e-2}` at `lr*`
- one cross term `(lr*/2, wd=1e-3)` to catch LR–WD interaction (in AdamW the
  effective per-step decay is `lr·wd`, so a strong-WD winner may prefer a
  lower LR)
- a **verbatim rerun of the Stage-A anchor** `(lr*, wd=0)` — the run-to-run
  noise bar that decides whether Stage B differences are real

**15 trials, ≈ 50 node-hours.**

Caveat to carry into analysis: on a 12K subset, overfitting appears ~9×
earlier than on the full set, so the subset will overstate how much WD
helps. Treat Stage B as establishing the *tolerance and trend* (does wd hurt
below some threshold? does λ_eff = lr·wd stay constant across widths?), and
confirm the absolute choice in Stage C. If a large wd wins on the subset,
prefer the largest wd that is *neutral-or-better* rather than the argmin.

## Stage C — full-scale confirmation

One run per width at the tuned `(lr*, wd*)`: full 111K dataset, production
config, **10–12 epoch budget** (~45 h W96, ~32 h W64 on GB200x4). Compare
val_loss at matched epochs against the incumbent curves, which already exist
at lr=1e-3/wd=0:

- W64: `mp-gga-ggau-width` run `zz3oecp7` (16-epoch best 0.008398)
- W96: run `z21di7sl` / `revived-energy-7` (ep10 0.008336 … ep27 0.006901)

W32 has no incumbent in this campaign; run it only if the trend fit needs
the third full-scale point.

## Analysis

- Fit `log lr*` vs `log width` → slope α; predict lr for W128/W160.
  (Expected α between 0 and −1.) Same check on wd: is `lr*·wd*`
  width-stable?
- Re-examine the width ranking at matched epochs with tuned recipes — this
  feeds the width-ablation writeup.
- All sweep runs log to W&B project **`mp-gga-ggau-lrwd`**. Run names are
  auto-generated `w{width}_{MMDD-HHMM}` (restart-segment convention), so
  aggregate by `config.lr` / `config.weight_decay` /
  `config.model.n_channels` — same pattern as
  `scripts/review_width_ablation.py`.

## Runbook

```bash
# 1. Get the production split files locally (small; either source works)
rclone copy cw:mp/chg_datasets/functionals/gga/split_capped.json /tmp/sweep/gga/
rclone copy cw:mp/chg_datasets/functionals/gga+u/split_capped.json /tmp/sweep/gga+u/
#   or: scp della:/scratch/gpfs/ROSENGROUP/common/globus_share_OA/mp/chg_datasets/functionals/gga/split_capped.json ...

# 2. Generate the sweep splits (checked into the branch for provenance)
uv run python scripts/coreweave/make_sweep_split.py \
    /tmp/sweep/gga/split_capped.json data/MP/sweep_splits/gga_split_sweep12k.json --frac 0.1112
uv run python scripts/coreweave/make_sweep_split.py \
    "/tmp/sweep/gga+u/split_capped.json" "data/MP/sweep_splits/gga+u_split_sweep12k.json" --frac 0.1112

# 3. Stage A configs (already generated in src/electrai/configs/MP/sweep_lrwd/)
uv run python scripts/coreweave/gen_lrwd_sweep.py --stage a
#    ... submit the printed submit.sh lines from a clean worktree; jobs run at
#    batch priority and won't starve the live W128 run.

# 4. After Stage A: regenerate for Stage B with the real per-width winners
uv run python scripts/coreweave/gen_lrwd_sweep.py --stage b \
    --best-lr 32=<lr*> 64=<lr*> 96=<lr*>
```

Each config filename stem doubles as the checkpoint namespace
(`run_training.sh` derives ckpt dirs from the stem), so trials never
cross-resume; the Stage-B noise rerun gets a `_rep2` stem for the same
reason.

## Known limitations

- 24K steps ≈ 0.9 production epochs per trial: LR ranking at short horizon
  with a completed cosine is a standard, usually-reliable proxy, but a
  near-tie between adjacent LRs at a width should be broken toward the
  *lower* LR (long-horizon optima drift down, not up).
- WD conclusions from the subset overstate regularization benefit (see
  Stage B caveat); Stage C is the arbiter.
- The scheduler steps per epoch, so an 8-epoch cosine has coarse resolution;
  all trials share it, so comparisons are fair.

## W128 withheld-test-set evaluation (Aug 5–7, 2026)

Checkpoint `w128_ckpt_epoch55_val0.005785.ckpt` (stage-C W128 at lr 2e-3,
epoch 55) evaluated on the withheld `test` split of `split_capped.json` —
never seen in training or validation. Della jobs 12057752 (GGA) + 12111587
(GGA+U), fp32, single A100-80GB; config
`src/electrai/configs/MP/config_gga_gga+u_w128_test_della.yaml`; per-sample
CSVs in `/scratch/gpfs/ROSENGROUP/bb9080/w128_test_eval/results{,_padsu}/`.

| Subset | n | mean NMAE | median | p90 | p99 | max | share <1% |
|---|---|---|---|---|---|---|---|
| GGA | 1700 | 0.480% | 0.373% | 0.900% | 1.842% | 2.95% | 92.0% |
| GGA+U (PADS) | 524 | 0.483% | 0.397% | 0.857% | 1.794% | 3.36% | 94.5% |
| **Combined** | **2224** | **0.481%** | 0.379% | 0.890% | 1.832% | 3.36% | 92.6% |

**Combined test NMAE 0.481% is under the 0.5% ChargE3Net threshold** and
consistent with the 0.579% val loss (val is bf16-on-GB200; test is fp32).

Provenance notes uncovered during this eval:

- The split files existed only in the buckets; they were staged from
  `s3://oa-electrai` to `/scratch/gpfs/ROSENGROUP/bb9080/w128_test_eval/`
  and their test/validation indices verified byte-identical to the
  training-time splits (via `data/MP/sweep_splits/*_sweep12k.json`
  pass-through).
- **The training data's `gga+u` inputs are PADS, not SAD.** Della's
  `functionals/gga+u_sad` and `gga+u_pads` share identical filelists and
  labels; only inputs differ. Evaluating with SAD inputs gives ~10% NMAE
  across the whole GGA+U test set (the model degrades SAD inputs below
  their own 7–9% input error); PADS inputs give 0.48%. Probe: Della job
  12111541. Any "GGA+U" result from the gga-ggau width campaign is a
  PADS-input result.
- bf16-mixed autocast OOMs this checkpoint on A100 (a ~75 GiB allocation
  inside a decoder conv on a 108³ sample, Della job 12049722) — evaluate at
  fp32 on Della, matching the throughput benchmark (job 12046200).
