#!/bin/bash

RUN_TAG="${1:-}"
if [ -z "$RUN_TAG" ]; then
    echo "Usage: $0 <RUN_TAG>" >&2
    exit 2
fi

: "${WATCH_INTERVAL:=60}"
: "${SAVE_EVERY:=200}"
: "${MAX_EPOCHS:=25}"
: "${FT_LOGGING:=/ibex/project/c2261/dac_iclr/finetune}"
: "${JOB_NAME:=ft_${RUN_TAG}}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${SHORT_SCRIPT:=$SCRIPT_DIR/run_finetune_mof_db_short.sh}"

LOG_DIR="$SCRIPT_DIR/output"
mkdir -p "$LOG_DIR"
WATCH_LOG="$LOG_DIR/watchdog_${RUN_TAG}.log"

SENTINEL="$FT_LOGGING/lightning_logs/$RUN_TAG/$RUN_TAG.done"

log() { printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$WATCH_LOG"; }

is_active() {
    # Returns 0 if a job named $JOB_NAME is queued/running for $USER.
    local out
    out=$(squeue -h -u "$USER" -n "$JOB_NAME" 2>/dev/null) || return 2
    [ -n "$out" ]
}

submit_one() {
    local out rc jobid
    out=$(RUN_TAG="$RUN_TAG" \
          SAVE_EVERY="$SAVE_EVERY" \
          MAX_EPOCHS="$MAX_EPOCHS" \
          sbatch --parsable --export=ALL \
                 --job-name="$JOB_NAME" \
                 "$SHORT_SCRIPT" 2>>"$WATCH_LOG")
    rc=$?
    jobid=$(printf '%s\n' "$out" | grep -Eo '^[0-9]+(\.[0-9]+)?$' | tail -n 1)
    if [ "$rc" -ne 0 ] || [ -z "$jobid" ]; then
        log "  sbatch failed (rc=$rc): $out"
        return 1
    fi
    log "  submitted jobid=$jobid"
    return 0
}

trap 'log "watchdog interrupted — exiting"; exit 0' INT TERM

log "=============================================================="
log "watchdog RUN_TAG=$RUN_TAG  JOB_NAME=$JOB_NAME"
log "WATCH_INTERVAL=${WATCH_INTERVAL}s  SAVE_EVERY=$SAVE_EVERY  MAX_EPOCHS=$MAX_EPOCHS"
log "short_script=$SHORT_SCRIPT"
log "sentinel=$SENTINEL"
log "=============================================================="

if [ ! -x "$SHORT_SCRIPT" ] && [ ! -r "$SHORT_SCRIPT" ]; then
    log "short script not found at $SHORT_SCRIPT — exiting"
    exit 2
fi

while true; do
    if [ -f "$SENTINEL" ]; then
        log "sentinel present — training reported done. exiting."
        break
    fi

    is_active
    rc=$?
    case "$rc" in
        0) log "job '$JOB_NAME' active — holding" ;;
        1) log "no active '$JOB_NAME' job — submitting"
           submit_one || log "  (will retry next tick)" ;;
        *) log "squeue failed (rc=$rc) — skipping tick" ;;
    esac

    sleep "$WATCH_INTERVAL"
done
