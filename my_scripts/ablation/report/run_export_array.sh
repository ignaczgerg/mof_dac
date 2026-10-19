#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name ablation_eval
#SBATCH -o output/eval.%A_%a.out
#SBATCH -e output/eval.%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1

set -e

conda activate aramco_dac
cd "$SLURM_SUBMIT_DIR"

: "${JOBLIST:?JOBLIST not set}"
: "${LOG_ROOT:=/ibex/project/c2261/dac_iclr/finetune/lightning_logs}"
: "${OUT_ROOT:=/ibex/project/c2261/dac_iclr/finetune/ablation_eval}"
: "${SPLITS:=test}"
: "${NUM_WORKERS:=4}"
: "${OVERWRITE:=0}"

LINE=$((SLURM_ARRAY_TASK_ID + 1))
RUN_TAG=$(sed -n "${LINE}p" "$JOBLIST" | awk '{print $1}')
CKPT=$(sed -n "${LINE}p" "$JOBLIST" | awk '{print $2}')

if [ -z "$RUN_TAG" ]; then
    echo "no run tag on line $LINE of $JOBLIST"
    exit 1
fi

hostname
nvidia-smi
echo "run_tag=$RUN_TAG"
echo "ckpt=$CKPT"

EXTRA=""
if [ "$OVERWRITE" = "1" ]; then
    EXTRA="--overwrite"
fi

python export_test_predictions.py \
    --run_tag "$RUN_TAG" \
    --ckpt "$CKPT" \
    --log_root "$LOG_ROOT" \
    --out_root "$OUT_ROOT" \
    --splits $SPLITS \
    --device cuda \
    --num_workers "$NUM_WORKERS" \
    $EXTRA
