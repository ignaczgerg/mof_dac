#!/bin/bash --login
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${FT_LOGGING:=/ibex/project/c2261/dac_iclr/finetune}"
: "${LOG_ROOT:=$FT_LOGGING/lightning_logs}"
: "${OUT_ROOT:=$FT_LOGGING/ablation_eval}"
: "${SPLITS:=test}"
: "${NUM_WORKERS:=4}"
: "${ARRAY_THROTTLE:=}"
: "${MANIFEST:=$SCRIPT_DIR/ablation_runs.tsv}"

OVERWRITE=0
AGGREGATE_ONLY=0
DRY_RUN=0
REPORT=""

usage() {
    cat <<USAGE
Usage: bash run_report.sh [options]

  --splits "test val"   splits to export (default: test)
  --report PATH         output .txt path (default: \$OUT_ROOT/ablation_metrics_<split>.txt)
  --out_root PATH       where predictions and the report are written
  --log_root PATH       lightning_logs root holding the ablation runs
  --overwrite           re-run inference even if predictions already exist
  --aggregate-only      skip inference, just rebuild the table from existing predictions
  --dry-run             print what would be submitted and exit
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --splits) SPLITS="$2"; shift 2 ;;
        --report) REPORT="$2"; shift 2 ;;
        --out_root) OUT_ROOT="$2"; shift 2 ;;
        --log_root) LOG_ROOT="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --aggregate-only) AGGREGATE_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

REPORT_SPLIT="$(echo "$SPLITS" | awk '{print $NF}')"
[ -n "$REPORT" ] || REPORT="$OUT_ROOT/ablation_metrics_${REPORT_SPLIT}.txt"

mkdir -p "$OUT_ROOT" output

find_ckpt() {
    tag="$1"
    for cand in \
        "$LOG_ROOT/$tag/aramco_ablation/$tag/checkpoints/last.ckpt" \
        "$LOG_ROOT/$tag/checkpoints/last.ckpt" \
        "$LOG_ROOT/$tag/checkpoints/step_latest.ckpt"; do
        if [ -f "$cand" ]; then
            echo "$cand"
            return 0
        fi
    done
    ls -t "$LOG_ROOT/$tag"/**/*.ckpt 2>/dev/null | head -1
}

if [ "$AGGREGATE_ONLY" = "1" ]; then
    conda activate aramco_dac
    python aggregate_metrics.py --out_root "$OUT_ROOT" --split "$REPORT_SPLIT" \
        --manifest "$MANIFEST" --report "$REPORT"
    exit 0
fi

JOBLIST="$OUT_ROOT/_joblist.txt"
: > "$JOBLIST"
MISSING=""
while IFS=$'\t' read -r RUN_ID RUN_TAG BLOCK LABEL; do
    [ -n "$RUN_TAG" ] || continue
    case "$RUN_ID" in \#*) continue ;; esac
    CKPT="$(find_ckpt "$RUN_TAG")"
    if [ -z "$CKPT" ]; then
        MISSING="$MISSING $RUN_ID"
        continue
    fi
    printf '%s\t%s\n' "$RUN_TAG" "$CKPT" >> "$JOBLIST"
done < "$MANIFEST"

N=$(wc -l < "$JOBLIST")
if [ "$N" -eq 0 ]; then
    echo "no checkpoints found under $LOG_ROOT" >&2
    exit 1
fi

echo "runs with checkpoints: $N"
if [ -n "$MISSING" ]; then
    echo "runs skipped (no checkpoint):$MISSING"
fi
echo "joblist: $JOBLIST"
echo "out_root: $OUT_ROOT"
echo "splits: $SPLITS"
echo "report: $REPORT"

if [ "$DRY_RUN" = "1" ]; then
    cat "$JOBLIST"
    exit 0
fi

ARRAY_SPEC="0-$((N - 1))"
if [ -n "$ARRAY_THROTTLE" ]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${ARRAY_THROTTLE}"
fi

EXPORT_ID=$(JOBLIST="$JOBLIST" LOG_ROOT="$LOG_ROOT" OUT_ROOT="$OUT_ROOT" SPLITS="$SPLITS" \
    NUM_WORKERS="$NUM_WORKERS" OVERWRITE="$OVERWRITE" \
    sbatch --parsable --array="$ARRAY_SPEC" \
    --export=ALL,JOBLIST,LOG_ROOT,OUT_ROOT,SPLITS,NUM_WORKERS,OVERWRITE \
    run_export_array.sh)
echo "submitted export array: $EXPORT_ID"

REPORT_ID=$(OUT_ROOT="$OUT_ROOT" SPLIT="$REPORT_SPLIT" REPORT="$REPORT" \
    sbatch --parsable --dependency=afterany:"$EXPORT_ID" \
    --export=ALL,OUT_ROOT,SPLIT,REPORT \
    run_aggregate.sh)
echo "submitted report job: $REPORT_ID (runs after $EXPORT_ID)"
echo "final table will be at: $REPORT"
