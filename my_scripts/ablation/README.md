# Ablation scripts — `adsorption_aramco`

One SLURM script per ablation run (see `../../ABLATION.md` for the full plan). The number
prefix is the **run order**. Submit from this directory so SLURM logs land in `output/`:

```bash
cd my_scripts/ablation
sbatch 1_baseline.sh      # run this first — it is the reference for blocks C-I
sbatch 2_gemnet_scratch.sh
...
```

Each script `cd ..` into `my_scripts/` and launches `finetune.py` on 4× V100. All runs share
`--seed 0` and log to the wandb project `aramco_ablation` (via `--wandb_project`).

| Order | Script | Run | What changes vs baseline |
|---|---|---|---|
| 1 | `1_baseline.sh` | A4 | baseline (EquiformerV2, ODAC, vanilla regression) |
| 2 | `2_gemnet_scratch.sh` | A1 | GemNet, from scratch |
| 3 | `3_gemnet_jmp.sh` | A2 | GemNet, JMP pretrained |
| 4 | `4_equiformer_jmp.sh` | A3 | EquiformerV2, JMP pretrained |
| 5 | `5_rbf_gaussian_scratch.sh` | B1 | RBF gaussian, scratch (block-B reference) |
| 6 | `6_rbf_bessel_scratch.sh` | B2 | RBF bessel, scratch |
| 7 | `7_rbf_coulomb_sturmian_scratch.sh` | B3 | RBF coulomb_sturmian, scratch |
| 8 | `8_heteroscedastic.sh` | C1 | heteroscedastic head |
| 9 | `9_classification_head.sh` | C2 | aux classification head on co2_uptake |
| 10 | `10_loss_huber.sh` | D2 | Huber loss |
| 11 | `11_loss_l2.sh` | D3 | L2 loss |
| 12 | `12_norm_log.sh` | E2 | log normalization |
| 13 | `13_uptake_per_cell.sh` | F2 | uptake conversion per_cell |
| 14 | `14_uptake_per_volume.sh` | F3 | uptake conversion per_volume |
| 15 | `15_roi_penalty.sh` | G2 | ROI penalty 2.0 |
| 16 | `16_invfreq.sh` | H2 | inverse-frequency reweighting |
| 17 | `17_feature_fusion.sh` | I2 | global MOF-descriptor feature fusion |

## LR sweep (run this before the ablations)

The LR-relevant optimizer/scheduler state comes from the backbone config builder
(`equiformer_v2_ft_config_` / `jmp_l_ft_config_`), **not** from the ablated flag — the
multitask builders override `config.optimizer` but never `config.lr_scheduler` or
`config.parameter_specific_optimizers`. So the runs group into 4 optimization regimes, and
13 of the 17 ablations share one:

| Regime | Covers | Grid |
|---|---|---|
| `eqv2_pt` | A3, A4, and all of blocks C–I (13 runs) | 3e-5, 1e-4, 3e-4 |
| `eqv2_scratch` | B1–B3 | 1e-4, 3e-4, 1e-3 |
| `gemnet_pt` | A2 | 3e-5, 1e-4, 3e-4 |
| `gemnet_scratch` | A1 | 1e-4, 3e-4, 1e-3 |

```bash
cd my_scripts/ablation
bash launch_sweep.sh                 # all 12, each under its own watchdog
bash launch_sweep.sh eqv2_scratch    # one regime's 3
bash sweep_watchdog_submit.sh eqv2_scratch 3.0e-4    # a single point
```

Each point runs **15 epochs on 35% of train** (`SWEEP_EPOCHS` / `SWEEP_FRAC` to change) and
requeues past the 4 h walltime under the same watchdog the ablations use, so it is a proper
run rather than a smoke test. Sweep run tags are `sweep_<regime>_lr<lr>`; the watchdog stops
on the `.done` sentinel like any other run. Pick the winner per regime by val `qst_co2_mae`
in wandb project `aramco_lr_sweep`, then edit `--lr` in the ablation scripts.

Warmup is set per regime (EquiformerV2 1 epoch, GemNet 3) to preserve each regime's real
warmup *fraction* at the shortened horizon — the real runs use 1/25 and 5/25 respectively, so
a shared warmup would change the schedule shape relative to the runs the LR is chosen for.

**Sweep results (2026-07-21, val `qst_co2_mae`, best epoch):**

| Regime | 3e-5 | 1e-4 | 3e-4 | 1e-3 | Winner |
|---|---|---|---|---|---|
| `eqv2_pt` | 0.556 | **0.545** | 0.605 | - | 1e-4 |
| `eqv2_scratch` | - | **0.806** | 0.852 | 1.117 | 1e-4 |
| `gemnet_pt` | 0.874 | **0.774** | 0.834 | - | 1e-4 |
| `gemnet_scratch` | - | 1.035 | **1.022** | NaN (diverged) | 3e-4 |

All ablation scripts keep `--lr 1.0e-4` except `2_gemnet_scratch.sh`, which uses `3.0e-4`.

Two caveats worth stating in the paper: the sweep selects on a 35% subset at a shortened
horizon, so it is a proxy, not the full-run optimum; and `eqv2_pt` is swept with
`odac_public`, with A3 (`jmp` weights, same architecture and schedule) inheriting that LR
rather than being swept separately.

## Comparison rules

- Blocks C–I: compare each run directly against `1_baseline.sh` (A4).
- Block A: compare within an architecture — GemNet {scratch vs JMP} (2 vs 3) and
  EquiformerV2 {JMP vs ODAC} (4 vs 1). Cross-architecture diffs confound architecture +
  pretraining.
- Block B (RBF) is scratch-only: compare 6/7 against the gaussian-scratch reference (5),
  **not** against the ODAC baseline.

## Resume

Each script sets a unique `--run_tag` and `--save_every_n_train_steps 500`. `finetune.py`
resumes from `<logging_path>/lightning_logs/<run_tag>/checkpoints/{last,step_latest}.ckpt`
when `--run_tag` is set, and writes `<run_tag>.done` after `trainer.fit` finishes.
`#SBATCH --requeue` covers preemption/node-failure; the **watchdog** covers walltime timeout.

## How to run (auto-requeue)

Each ablation needs more than the 4 h SLURM walltime, so a watchdog resubmits the job after
every timeout until the `.done` sentinel appears (resuming from the latest checkpoint each
time). `FT_LOGGING` must equal the runs' `--logging_path` so the watchdog finds the sentinel.

**All 17 (recommended)** — submits the baseline, waits for the shared `data_cache` to warm,
then launches the rest so they reuse the cached filtering/sizes/stats:

```bash
cd my_scripts/ablation
bash launch_all.sh                 # or: nohup bash launch_all.sh &   (it idles ~15 min while warming)
```

**One run:**

```bash
cd my_scripts/ablation
bash watchdog_submit.sh 8_heteroscedastic.sh
```

Each watchdog keeps exactly one job named `ft_<RUN_TAG>` alive (it holds while the job is
queued/running, resubmits when none is active). Use the watchdog **or** a manual `sbatch` for a
given run, not both (a manual submit has a different job name and the watchdog would add a
duplicate).

Monitor / stop:

```bash
squeue -u $USER                                   # queued/running jobs
tail -f output/watchdog_ablation_*.out            # watchdog activity
pkill -f run_watchdog_finetune.sh                 # stop all watchdogs (jobs keep running)
```

A watchdog resubmits a job that *fails* immediately too, so it will loop on a genuinely broken
run — watch the first cycle of any new run, and `touch <logging_path>/lightning_logs/<run_tag>/<run_tag>.done`
to force-stop a watchdog if needed.

## Test-set report (r2 / pearson / spearman / rmse / mae)

The training runs never ran a test pass, so the test metrics are produced by re-running
inference from each run's checkpoint and scoring the dumped per-structure predictions.
Everything lives in `report/`:

| File | What it does |
|---|---|
| `report/ablation_runs.tsv` | run id / run tag / block / table label - the single place to edit if runs change |
| `report/export_test_predictions.py` | rebuilds the model+config from one checkpoint and writes `sid/gt/pred` npz for a split |
| `report/run_export_array.sh` | SLURM array wrapper, one 1x V100 task per run |
| `report/aggregate_metrics.py` | reads the npz files and writes the LaTeX table (`.txt`) plus a raw `.csv` |
| `report/run_aggregate.sh` | SLURM wrapper for the aggregation step |
| `report/run_report.sh` | driver: submits the array, then the aggregation job with `--dependency=afterany` |

```bash
cd my_scripts/ablation/report
bash run_report.sh                  # export all runs, then build the table
bash run_report.sh --dry-run        # show which runs have checkpoints, submit nothing
bash run_report.sh --aggregate-only # rebuild the table from existing predictions
bash run_report.sh --splits "val test" --overwrite
```

The array runs unthrottled by default (one task per run, all at once); set
`ARRAY_THROTTLE=4` to cap concurrency.

Output (default `$FT_LOGGING/ablation_eval`):

```
ablation_eval/<run_tag>/predictions/final/test/adsorption_aramco/{qst_co2,co2_uptake}.npz
ablation_eval/ablation_metrics_test.txt        <- copy-pastable LaTeX tabular (one per target)
ablation_eval/ablation_metrics_test.csv        <- same numbers, unrounded
ablation_eval/ablation_predictions_test.csv.gz <- every per-structure sid/gt/pred, all runs
```

The per-structure predictions are kept, so any other metric can be computed later without
re-running inference: read the npz files directly (`sid`, `gt`, `pred`, plus `std` for the
heteroscedastic run) or the consolidated `ablation_predictions_test.csv.gz`
(`run_id, run_tag, target, split, sid, gt, pred`). `--aggregate-only` rebuilds the table from
them in seconds. The export step also skips any run whose npz files already exist unless
`--overwrite` is passed.

Checkpoint preference per run: `aramco_ablation/<tag>/checkpoints/last.ckpt`, then
`checkpoints/last.ckpt`, then `checkpoints/step_latest.ckpt`. Runs without a checkpoint are
skipped and appear as `--` rows with a `% missing predictions:` note under the table.

Notes:

- Inference is single-GPU on purpose. `_should_log_now()` gates prediction dumping to rank 0,
  so a DDP pass would silently save only one rank's shard.
- The saved `gt`/`pred` are already denormalized: standard/log normalization is inverted and
  `per_cell` / `per_volume` uptake is converted back, so F2/F3 are comparable to the baseline.
- The `log` normalization run (E2) stores `qst_co2` with the opposite sign (the log transform
  works on the magnitude). All five metrics are invariant to negating both `gt` and `pred`, so
  the row is still comparable; only the raw npz values look flipped.
- The heteroscedastic run (C1) additionally stores `std` in the npz; the report ignores it.
- Predictions are computed on the same `max_natoms 700` filtered test split for every run
  (12,196 of 12,212 structures), so the rows are row-comparable.
