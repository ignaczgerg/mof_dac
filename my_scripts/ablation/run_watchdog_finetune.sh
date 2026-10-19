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
: "${MIN_HEALTHY_RUNTIME:=300}"
: "${MAX_CONSECUTIVE_FAILURES:=5}"

if [ -z "${SHORT_SCRIPT:-}" ]; then
    echo "ERROR: SHORT_SCRIPT must be set to the sbatch script to (re)submit (use watchdog_submit.sh or sweep_watchdog_submit.sh)" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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
    LAST_JOBID="$jobid"
    return 0
}

# Elapsed comes back as [DD-][HH:]MM:SS; normalize to seconds.
elapsed_to_seconds() {
    local e="$1" days=0 h=0 m=0 s=0
    case "$e" in
        *-*) days="${e%%-*}"; e="${e#*-}" ;;
    esac
    case "$e" in
        *:*:*) IFS=: read -r h m s <<<"$e" ;;
        *:*)   IFS=: read -r m s <<<"$e" ;;
        *)     s="$e" ;;
    esac
    printf '%s' $(( 10#${days:-0} * 86400 + 10#${h:-0} * 3600 + 10#${m:-0} * 60 + 10#${s:-0} ))
}

# Judge the previous job by how long it actually RAN (sacct Elapsed), not by
# wall-clock since submit -- a job can sit PENDING for ages and then die in 30s.
# Returns: 0 healthy, 1 died young, 2 unknown/not-yet-recorded.
check_last_job() {
    [ -z "$LAST_JOBID" ] && return 2
    command -v sacct >/dev/null 2>&1 || return 2

    local row state elapsed secs
    row=$(sacct -j "$LAST_JOBID" -n -X -P -o State,Elapsed 2>/dev/null | head -n 1) || return 2
    [ -z "$row" ] && return 2

    state="${row%%|*}"
    elapsed="${row##*|}"
    state="${state%% *}"

    case "$state" in
        PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|"") return 2 ;;
    esac

    secs=$(elapsed_to_seconds "$elapsed")
    if [ "$secs" -lt "$MIN_HEALTHY_RUNTIME" ]; then
        log "  job $LAST_JOBID: state=$state elapsed=${elapsed} (${secs}s) — died young"
        return 1
    fi
    log "  job $LAST_JOBID: state=$state elapsed=${elapsed} (${secs}s) — ran long enough"
    return 0
}

trap 'log "watchdog interrupted — exiting"; exit 0' INT TERM

LAST_JOBID=""
CONSECUTIVE_FAILURES=0

log "=============================================================="
log "watchdog RUN_TAG=$RUN_TAG  JOB_NAME=$JOB_NAME"
log "WATCH_INTERVAL=${WATCH_INTERVAL}s  SAVE_EVERY=$SAVE_EVERY  MAX_EPOCHS=$MAX_EPOCHS"
log "give up after $MAX_CONSECUTIVE_FAILURES jobs running <${MIN_HEALTHY_RUNTIME}s"
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
           check_last_job
           case "$?" in
               0) CONSECUTIVE_FAILURES=0 ;;
               1) CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
                  log "  consecutive short-lived jobs: $CONSECUTIVE_FAILURES/$MAX_CONSECUTIVE_FAILURES" ;;
           esac

           if [ "$CONSECUTIVE_FAILURES" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
               log "$MAX_CONSECUTIVE_FAILURES consecutive jobs died in <${MIN_HEALTHY_RUNTIME}s — this is a crash loop, not a walltime requeue."
               log "giving up. inspect $LOG_DIR/gpu.${LAST_JOBID}.err before resubmitting."
               exit 1
           fi

           submit_one || log "  (will retry next tick)" ;;
        *) log "squeue failed (rc=$rc) — skipping tick" ;;
    esac

    sleep "$WATCH_INTERVAL"
done
