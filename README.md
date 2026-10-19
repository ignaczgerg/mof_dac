# Aramco DAC: CO2 Adsorption Ablation Study

This repository contains the code for an ablation study on fine-tuning pretrained
atomistic foundation models for CO2 adsorption property prediction in metal-organic
frameworks (MOFs). It builds on JMP (Joint Multi-domain Pre-training) and fine-tunes
GemNet and EquiformerV2 backbones on the `adsorption_aramco` MOF dataset for two targets:

- `qst_co2`: isosteric heat of CO2 adsorption
- `co2_uptake`: CO2 uptake

The study is a one-factor-at-a-time ablation (17 runs) relative to an EquiformerV2 / ODAC
baseline, covering pretraining source, RBF basis, head strategy, loss, normalization,
uptake conversion, ROI penalty, inverse-frequency reweighting, and feature fusion. The
best configuration (F2, `per_cell` uptake conversion) is also used for a large-scale CIF
screening campaign.

## Repository layout

```
ABLATION.md                         full ablation plan, run matrix, comparison rules
paths.yaml                          environment-specific data / checkpoint / logging paths
environment.yml, pyproject.toml     conda environment and the `jmp` package
src/jmp/                            models, datasets, fine-tuning tasks, Lightning utilities
my_scripts/finetune.py              fine-tuning entry point
my_scripts/setup_finetune.py        config builders and checkpoint / dataset resolution
my_scripts/ablation/                one SLURM script per ablation run, LR sweep, watchdog
my_scripts/ablation/report/         test-set export and metric table
my_scripts/inference/mof_transformer_campaign/   F2 screening campaign
```

## Installation

```bash
conda env create -f environment.yml
conda activate aramco_dac
pip install -e .
```

The environment pins PyTorch with CUDA 12.1, PyG and Lightning 2.2.1. Training and
inference require NVIDIA GPUs; the provided SLURM scripts target V100 nodes.

## Configuring paths

`paths.yaml` defines one block per environment under `environments`:

```yaml
environments:
  ibex:
    root: /path/to/project_root
    ocp: /path/to/ocp_data
    pt_logging: /path/to/pretrain_logs
    ft_logging: /path/to/finetune_logs
```

The environment is chosen automatically by `jmp.utils.env_utils.detect_env()` (`ibex` if
`/ibex` exists, otherwise `docker`). Edit the block that matches your machine:

- `root`: must contain the pretrained checkpoints, for example
  `checkpoints/JMP/jmp-s.pt` (GemNet) and `checkpoints/EquiformerV2/eqv2_31M_odac.pt`
  (EquiformerV2, ODAC). The full tag-to-file mapping is `CHECKPOINT_TAG_MAPPING` in
  `my_scripts/setup_finetune.py`.
- `ft_logging`: default `--logging_path` for fine-tuning runs (checkpoints, `data_cache`,
  `.done` sentinels).

The `adsorption_aramco` dataset location is set in `my_scripts/setup_finetune.py`; update it
to point at your copy of the dataset. The SLURM scripts also default `FT_LOGGING`, the
campaign directories and the checkpoint paths to cluster locations; override them through
the environment variables at the top of each script.

## Running the ablation

The full plan (baseline command, run matrix, seeding, comparison rules) is in
[ABLATION.md](ABLATION.md). Script-level details (run order, LR sweep, resume, monitoring)
are in [my_scripts/ablation/README.md](my_scripts/ablation/README.md).

All runs use `--tasks adsorption_aramco --targets qst_co2 co2_uptake` with `--seed 0` and
log to the wandb project `aramco_ablation`.

1. Optional LR sweep (one grid per optimization regime):

   ```bash
   cd my_scripts/ablation
   bash launch_sweep.sh
   ```

2. Launch all 17 ablation runs. Each run exceeds the 4 h walltime, so every run is kept
   alive by a watchdog (`run_watchdog_finetune.sh`) that resubmits it after each timeout
   and resumes from the latest checkpoint until the `<run_tag>.done` sentinel appears:

   ```bash
   cd my_scripts/ablation
   bash launch_all.sh                            # baseline first, then the other 16
   bash watchdog_submit.sh 8_heteroscedastic.sh  # or a single run
   ```

   `FT_LOGGING` must match the runs' `--logging_path` so the watchdog finds the sentinel.

## Test-set report

The training runs do not run a test pass. `my_scripts/ablation/report/` re-runs
single-GPU inference from each run's checkpoint, stores per-structure `sid/gt/pred`
predictions, and aggregates r2 / pearson / spearman / rmse / mae into a LaTeX table and CSV.

```bash
cd my_scripts/ablation/report
bash run_report.sh                  # export all runs, then build the table
bash run_report.sh --dry-run        # list runs with checkpoints, submit nothing
bash run_report.sh --aggregate-only # rebuild the table from existing predictions
```

Runs are listed in `report/ablation_runs.tsv`. Output goes to `$FT_LOGGING/ablation_eval`
by default.

## F2 screening campaign

`my_scripts/inference/mof_transformer_campaign/` runs the F2 (`per_cell`) checkpoint over a
large set of CIF files, sharded across a SLURM array (one V100 per shard).

1. Write `$CAMPAIGN_DIR/manifest.txt`: the first line is the CIF directory, each following
   line is a CIF file name.
2. Run the shard array (one task per `SHARD_SIZE` files, default 7500). Set `CKPT` to the F2
   checkpoint and `CODE_DIR` to this repository's `src` directory:

   ```bash
   cd my_scripts/inference/mof_transformer_campaign
   CAMPAIGN_DIR=/path/to/campaign CKPT=/path/to/last.ckpt CODE_DIR=$PWD/../../../src \
       sbatch --array=0-149 --export=ALL run_shard_array_ablation.sh
   ```

   Each shard writes `shards/shard_XXXX_of_NNNN/` with per-target `.npz` predictions, a
   `shard_meta.json`, and a `DONE` marker; finished shards are skipped on resubmission.
   Setting `VALIDATE_LMDB` instead checks the CIF pipeline against the ablation test
   predictions.

3. Merge and verify once all shards are done:

   ```bash
   CAMPAIGN_DIR=/path/to/campaign NUM_SHARDS=150 sbatch --export=ALL run_merge.sh
   ```

   `run_merge.sh` refuses to merge an incomplete campaign, merges the shards with
   `merge_shards.py` into `$CAMPAIGN_DIR/merged/`, and writes `verification.json`.

## Acknowledgements

The model and training code in `src/jmp` is derived from the JMP codebase
(Joint Multi-domain Pre-training) and uses GemNet and EquiformerV2 backbones.
