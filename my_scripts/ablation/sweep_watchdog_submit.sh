#!/bin/bash
# Start the auto-requeue watchdog for ONE sweep point.
#
# Same mechanism as watchdog_submit.sh, but the sweep job script is parametrized
# by env rather than carrying a literal RUN_TAG= line, so the tag is computed
# here and handed to the watchdog directly. run_watchdog_finetune.sh resubmits
# with `sbatch --export=ALL`, which carries SWEEP_* through every requeue.
#
# Usage:  bash sweep_watchdog_submit.sh <regime> <lr>
# Env:    FT_LOGGING (must equal the run's --logging_path; default below)
#         SWEEP_EPOCHS (default 15), SWEEP_FRAC (default 0.35)
#         SAVE_EVERY (default 500), WATCH_INTERVAL (poll seconds, default 60)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WD="$SCRIPT_DIR/run_watchdog_finetune.sh"
OUT="$SCRIPT_DIR/output"
: "${FT_LOGGING:=/ibex/project/c2261/dac_iclr/finetune}"
: "${SWEEP_EPOCHS:=15}"
: "${SWEEP_FRAC:=0.35}"
: "${SAVE_EVERY:=500}"

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <regime> <lr>" >&2
    exit 2
fi
REGIME="$1"
LR="$2"

case "$REGIME" in
    eqv2_pt|eqv2_scratch|gemnet_pt|gemnet_scratch) ;;
    *) echo "ERROR: unknown regime '$REGIME'" >&2; exit 2 ;;
esac

RUN_TAG="sweep_${REGIME}_lr${LR}"

mkdir -p "$OUT"
# Submit with CWD inside ablation/ so the job's `cd ..` lands in my_scripts/ and
# its #SBATCH -o output/... lands in ablation/output/.
cd "$SCRIPT_DIR"
SHORT_SCRIPT="$SCRIPT_DIR/sweep_lr_job.sh" \
    FT_LOGGING="$FT_LOGGING" \
    SWEEP_REGIME="$REGIME" SWEEP_LR="$LR" \
    SWEEP_EPOCHS="$SWEEP_EPOCHS" SWEEP_FRAC="$SWEEP_FRAC" \
    SAVE_EVERY="$SAVE_EVERY" \
    nohup bash "$WD" "$RUN_TAG" > "$OUT/watchdog_${RUN_TAG}.out" 2>&1 &
echo "started watchdog pid=$! run_tag=$RUN_TAG regime=$REGIME lr=$LR"
echo "  log: $OUT/watchdog_${RUN_TAG}.out"
