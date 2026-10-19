# Ablation Study — `adsorption_aramco`

Single-dataset, one-factor-at-a-time (linear) ablation. Every run changes **exactly one**
thing relative to the baseline.

This ablation needs to run in a new WANDB project: aramco_ablation

## Harness decision (important)

**All runs use `--tasks adsorption_aramco` and DO NOT pass `--dataset_name`.**

Reason: `heteroscedastic`, `enable_classification`, and `enable_feature_fusion` are only
wired into the multitask config builder (`equiformer_v2_adsorption_db_multitask_config_` /
`jmp_l_adsorption_db_multi_task_config_`). The single-task `--dataset_name` path silently
ignores those flags. Passing `--dataset_name` forces the single-task branch
(`setup_finetune.py:220, 328`), so we leave it unset and drive everything through `--tasks`
with a single task. This keeps the harness identical across all runs (same `primary_metric`,
same metrics) so every comparison differs in one variable only.

`roi_penalty`, `invfreq`, `loss`, `normalization_type`, `uptake_conversion_mode`,
`graph_scalar_reduction` are read directly from `args` and work on this path.

> adabin is **dropped** — not needed for a single dataset, and it only exists on the
> `--dataset_name` path anyway.

## Targets

Two targets only: **`qst_co2`** and **`co2_uptake`**. (Loss coefficients `0.5 0.5`.)
`kh_co2` is hard-excluded for aramco and is not used.

## Baseline (reference run)

| Setting | Value |
|---|---|
| Backbone / checkpoint | EquiformerV2 (`--small`), ODAC pretrained (`--checkpoint_tag odac_public`) |
| Model size | small (`--small`) — **required**; without a size flag `config.backbone` is never built (`equiformer_v2.py:71-77`) and the run crashes. `--small` + ODAC maps to `small_odac()`. |
| Task head | vanilla scalar regression |
| Loss | L1 (`--loss l1`) |
| Normalization | standard (`--normalization_type standard standard`) |
| Uptake conversion | none |
| ROI penalty | 1.0 (disabled) |
| Inverse-frequency | off |
| Feature fusion | off |
| Graph scalar reduction | mean (`--graph_scalar_reduction mean mean`) |
| Loss coefficients | `0.5 0.5` |

The baseline run is **run A4** below. It is reused as the reference for every block —
don't run it more than once.

## Baseline command

Run from `my_scripts/` (the SLURM wrapper does `cd ..` into `my_scripts` before calling
`finetune.py`; see template at the bottom).

```bash
python finetune.py \
    --tasks adsorption_aramco \
    --targets qst_co2 co2_uptake \
    --targets_loss_coefficients 0.5 0.5 \
    --model_name equiformer_v2 \
    --small \
    --checkpoint_tag odac_public \
    --loss l1 \
    --normalization_type standard standard \
    --graph_scalar_reduction mean mean \
    --uptake_conversion_mode none \
    --roi_penalty 1.0 \
    --lr 1.0e-4 \
    --epochs 25 \
    --weight_decay 5e-3 \
    --seed 0 \
    --num_workers 4 \
    --atom_bucket_batch_sampler \
    --max_atoms_per_batch 700 \
    --max_natoms 700 \
    --cutoff 6 \
    --max_neighbors 30 \
    --rbf_function gaussian \
    --num_distance_basis 600 \
    --val_same_as_train \
    --enable_wandb
```

Each run below = **baseline command + the override(s) in the "Flag override" column.**

## Seeding (reproducibility)

All runs pin `--seed 0` so every comparison differs in the ablated variable only, not in
init/shuffle noise. `--seed` is fully wired: each backbone config sets
`config.trainer.seed = args.seed` (`equiformer_v2.py:56,369`, `jmp_l.py:44`), which Lightning
consumes via `seed_context` (`trainer.py:211-214`) to seed model init, data shuffling, and
workers; the same `args.seed` also drives dataset subsampling (`DatasetSampleNConfig.seed`).
Default is `0`, so a single `--seed` value controls the whole pipeline.

All runs use the single fixed seed `--seed 0`. Note this means single-run-per-config with
**no error bars** — most relevant for the seed-sensitive from-scratch runs (A1, B1–B3), whose
differences should be read as directional rather than statistically separated.

---

## Run matrix (17 runs)

### Block A — Backbone / pretraining (vanilla regression, all else baseline)

GemNet pretraining = JMP (`jmp-s.pt`). EquiformerV2 pretraining = JMP or ODAC.

| # | Run | Flag override |
|---|---|---|
| A1 | GemNet, scratch | `--model_name gemnet --scratch` *(remove `--checkpoint_tag`, `--rbf_function`, `--num_distance_basis`)* |
| A2 | GemNet, JMP | `--model_name gemnet` *(remove `--checkpoint_tag`, `--rbf_function`, `--num_distance_basis`)* |
| A3 | EquiformerV2, JMP | `--checkpoint_tag jmp` |
| **A4** | **EquiformerV2, ODAC** | **(= baseline, no override)** |

> **Why GemNet runs must drop `--rbf_function` / `--num_distance_basis`:** after
> `config.finalize()`, `finetune.py:491-495` does
> `config.backbone.distance_function = args.rbf_function` and
> `config.backbone.num_distance_basis = ...`. GemNet's `GOCBackboneConfig` has no such
> fields, so setting them on the finalized (frozen) config raises. Both assignments are
> guarded by truthiness (`if hasattr(args, "rbf_function") and args.rbf_function:` /
> `... is not None:`), so **omitting** the flags (defaults `None`) is what makes the guards
> skip — do not pass them for GemNet. `--checkpoint_tag` is likewise EquiformerV2-only
> (GemNet hardcodes `checkpoints/JMP/jmp-s.pt`, `setup_finetune.py:200-203`).

### Block B — RBF basis (EquiformerV2, from scratch)

Changing the RBF basis / distance basis changes backbone tensor shapes, so these **cannot**
load a pretrained checkpoint → `--scratch`. Compare only within this block (scratch
reference), not against the ODAC baseline.

| # | Run | Flag override |
|---|---|---|
| B1 | gaussian (scratch) | `--scratch --rbf_function gaussian` *(remove `--checkpoint_tag`)* |
| B2 | bessel (scratch) | `--scratch --rbf_function bessel` *(remove `--checkpoint_tag`)* |
| B3 | coulomb_sturmian (scratch) | `--scratch --rbf_function coulomb_sturmian` *(remove `--checkpoint_tag`)* |

### Block C — Head strategy (ODAC baseline + toggle)

| # | Run | Flag override |
|---|---|---|
| C1 | Heteroscedastic | `--heteroscedastic` *(uses default `hetero_nll_weight=1.0`, `hetero_std_weight=0.1`, `hetero_min_var=1e-6`)* |
| C2 | Classification head | `--enable_classification --classification_targets co2_uptake --classification_threshold 0.5 --classification_loss_weight 1.0 --focal_gamma 2.0 --focal_alpha_pos 0.75 --focal_alpha_neg 0.25` |

### Block D — Loss function

| # | Run | Flag override |
|---|---|---|
| D1 | L1 | (= baseline) |
| D2 | Huber | `--loss huber` |
| D3 | L2 | `--loss l2` |

### Block E — Normalization

| # | Run | Flag override |
|---|---|---|
| E1 | standard | (= baseline) |
| E2 | log | `--normalization_type log log` |

### Block F — Uptake conversion mode

Only affects `co2_uptake` (an uptake target); `qst_co2` is unaffected.

| # | Run | Flag override |
|---|---|---|
| F1 | none | (= baseline) |
| F2 | per_cell | `--uptake_conversion_mode per_cell` |
| F3 | per_volume | `--uptake_conversion_mode per_volume` |

### Block G — ROI loss penalty

| # | Run | Flag override |
|---|---|---|
| G1 | off (roi=1.0) | (= baseline) |
| G2 | roi penalty | `--roi_penalty 2.0` |

### Block H — Inverse-frequency reweighting

| # | Run | Flag override |
|---|---|---|
| H1 | off | (= baseline) |
| H2 | invfreq | `--invfreq --invfreq-bins 50 --invfreq-beta 0.75 --invfreq-warmup-epochs 2` |

### Block I — Auxiliary objective: feature fusion

Injects global MOF descriptors (PLD/LCD/…). Requires those descriptor fields to exist in the
aramco LMDBs; geometric features that are also targets are auto-stripped.

| # | Run | Flag override |
|---|---|---|
| I1 | no fusion | (= baseline) |
| I2 | feature fusion | `--enable_feature_fusion --fusion_feature_names pld lcd gcd unitcell_volume density asa av nav --fusion_type early` |

---

## Unique runs to launch

Baseline (A4) is shared, so the distinct runs are:

```
A1, A2, A3, A4(baseline), B1, B2, B3, C1, C2, D2, D3, E2, F2, F3, G2, H2, I2
```

= **17 runs.** (D1/E1/F1/G1/H1/I1 are the baseline, not re-run.)

## Comparison rules

- **Blocks C–I** compare each toggle directly against baseline **A4** (single variable).
- **Block A** is the pretraining-source comparison: GemNet {scratch vs JMP} and
  EquiformerV2 {JMP vs ODAC}. Compare *within an architecture* — GemNet-vs-Equiformer
  differences confound architecture + pretraining and are not pretraining ablations.
- **Block B** (RBF) is scratch-only; compare within the block against the gaussian-scratch
  reference (B1), **not** against the ODAC baseline.

---

## SLURM template (4× V100, 4 h, auto-resubmission)

Resume is supported via `--run_tag` (resolves `last.ckpt` / `step_latest.ckpt`),
`--save_every_n_train_steps`, and a `.done` sentinel; pair with
`my_scripts/ablation/run_watchdog_finetune.sh` for auto-resubmission.

```bash
#!/bin/bash --login
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=350G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
conda activate aramco_dac
cd ..

RUN_TAG="ablation_A4_baseline"   # unique per run; enables resume

srun --cpu-bind=none python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv-conf timeout=300 \
    finetune.py \
    --tasks adsorption_aramco \
    --targets qst_co2 co2_uptake \
    --targets_loss_coefficients 0.5 0.5 \
    --model_name equiformer_v2 \
    --small \
    --checkpoint_tag odac_public \
    --loss l1 \
    --normalization_type standard standard \
    --graph_scalar_reduction mean mean \
    --uptake_conversion_mode none \
    --roi_penalty 1.0 \
    --lr 1.0e-4 --epochs 25 --weight_decay 5e-3 --seed 0 --num_workers 4 \
    --atom_bucket_batch_sampler --max_atoms_per_batch 700 --max_natoms 700 \
    --cutoff 6 --max_neighbors 30 \
    --rbf_function gaussian --num_distance_basis 600 \
    --val_same_as_train \
    --run_tag "$RUN_TAG" --save_every_n_train_steps 500 \
    --enable_wandb
    # per-run: append the "Flag override" for this run here
```

Give each run a unique `--run_tag` (e.g. `ablation_C1_hetero`) so resume/checkpoints don't
collide.
