#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name ablation_report
#SBATCH -o output/report.%A.out
#SBATCH -e output/report.%A.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1

set -e

conda activate aramco_dac
cd "$SLURM_SUBMIT_DIR"

: "${OUT_ROOT:=/ibex/project/c2261/dac_iclr/finetune/ablation_eval}"
: "${SPLIT:=test}"
: "${REPORT:=}"

ARGS=(--out_root "$OUT_ROOT" --split "$SPLIT")
if [ -n "$REPORT" ]; then
    ARGS+=(--report "$REPORT")
fi

python aggregate_metrics.py "${ARGS[@]}"
