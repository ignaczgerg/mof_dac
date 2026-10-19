#!/bin/bash
# Start the auto-requeue watchdog for ONE ablation script.
# The watchdog (run_watchdog_finetune.sh) keeps a job named ft_<RUN_TAG>
# alive: it (re)submits the script whenever no such job is queued/running, and
# stops once finetune.py writes <FT_LOGGING>/lightning_logs/<RUN_TAG>/<RUN_TAG>.done.
#
# Usage:  bash watchdog_submit.sh <ablation_script.sh>
# Env:    FT_LOGGING  (must equal the run's --logging_path; default below)
#         WATCH_INTERVAL (poll seconds, default 60)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WD="$SCRIPT_DIR/run_watchdog_finetune.sh"
OUT="$SCRIPT_DIR/output"
: "${FT_LOGGING:=/ibex/project/c2261/dac_iclr/finetune}"

SCRIPT="$(realpath "$1")"
RUN_TAG="$(sed -n 's/^[[:space:]]*RUN_TAG="\([^"]*\)".*/\1/p' "$SCRIPT" | head -1)"
if [ -z "$RUN_TAG" ]; then
    echo "ERROR: could not parse RUN_TAG from $SCRIPT" >&2
    exit 2
fi

mkdir -p "$OUT"
# Submit with CWD inside my_scripts/ so the ablation script's `cd ..` lands there.
cd "$ROOT/my_scripts/ablation"
SHORT_SCRIPT="$SCRIPT" FT_LOGGING="$FT_LOGGING" \
    nohup bash "$WD" "$RUN_TAG" > "$OUT/watchdog_${RUN_TAG}.out" 2>&1 &
echo "started watchdog pid=$! run_tag=$RUN_TAG script=$SCRIPT"
echo "  log: $OUT/watchdog_${RUN_TAG}.out"
