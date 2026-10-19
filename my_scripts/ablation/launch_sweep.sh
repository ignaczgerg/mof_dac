#!/bin/bash
# Launch the LR sweep: 4 optimization regimes x 3 learning rates = 12 runs,
# each under its own auto-requeue watchdog.
#
# Why per-regime and not per-ablation: the LR-relevant optimizer/scheduler state is
# set by the backbone config builder (equiformer_v2_ft_config_ / jmp_l_ft_config_),
# not by the ablated flag. Runs sharing a backbone+pretraining setting share a
# regime, so 13 of the 17 ablations (A3, A4, and all of blocks C-I) inherit one LR.
#
# Each point runs SWEEP_EPOCHS on SWEEP_FRAC of train and requeues past the 4 h
# walltime until it finishes, same as the ablations. Pick the winner per regime by
# val qst_co2_mae in wandb (project: aramco_lr_sweep), then hardcode it into the
# ablation scripts.
#
# Usage:  bash launch_sweep.sh              # all 12
#         bash launch_sweep.sh eqv2_scratch # one regime's 3
# Env:    FT_LOGGING (must equal the runs' --logging_path; default below)
#         SWEEP_EPOCHS (default 15), SWEEP_FRAC (default 0.35)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${FT_LOGGING:=/ibex/project/c2261/dac_iclr/finetune}"
: "${SWEEP_EPOCHS:=15}"
: "${SWEEP_FRAC:=0.35}"
export FT_LOGGING SWEEP_EPOCHS SWEEP_FRAC

# Pretrained regimes bracket the current 1e-4; scratch regimes bracket upward,
# since EquiformerV2's reference config trains from scratch at 4e-4.
LRS_PRETRAINED="3.0e-5 1.0e-4 3.0e-4"
LRS_SCRATCH="1.0e-4 3.0e-4 1.0e-3"

submit_regime () {
    local regime="$1" lrs
    case "$regime" in
        eqv2_pt|gemnet_pt)           lrs="$LRS_PRETRAINED" ;;
        eqv2_scratch|gemnet_scratch) lrs="$LRS_SCRATCH" ;;
        *) echo "ERROR: unknown regime '$regime'" >&2; return 2 ;;
    esac
    for lr in $lrs; do
        bash "$HERE/sweep_watchdog_submit.sh" "$regime" "$lr"
        sleep 2
    done
}

if [ "$#" -ge 1 ]; then
    submit_regime "$1"
    exit $?
fi

# Submit one point first and wait for the shared data_cache to warm, so the other
# 11 reuse the sizes/stats instead of each recomputing them.
#
# The sizes/stats cache key includes sample_n, so SWEEP_FRAC creates a NEW entry
# even though the filtered-indices entry is already warm from the ablations. Probe
# for a sizes.npy written AFTER this launcher started -- an unqualified existence
# check would match the ablations' entries and return immediately.
STAMP="$(mktemp)"
trap 'rm -f "$STAMP"' EXIT

echo "== submitting first point (warms data_cache for frac=$SWEEP_FRAC) =="
bash "$HERE/sweep_watchdog_submit.sh" eqv2_pt 1.0e-4

echo "== waiting for a fresh sizes.npy under $FT_LOGGING/data_cache (max ~40 min) =="
warm=0
for i in $(seq 1 160); do
    if find "$FT_LOGGING/data_cache" -name 'sizes.npy' -newer "$STAMP" 2>/dev/null | grep -q .; then
        echo "   cache warm after ~$(( i * 15 ))s"; warm=1; break
    fi
    sleep 15
done
[ "$warm" -eq 0 ] && echo "   WARN: cache not detected warm; launching anyway (first wave may pay full load)."

echo "== submitting remaining 11 points =="
for lr in 3.0e-5 3.0e-4; do
    bash "$HERE/sweep_watchdog_submit.sh" eqv2_pt "$lr"
    sleep 2
done
for regime in eqv2_scratch gemnet_pt gemnet_scratch; do
    submit_regime "$regime"
done

echo "== 12 sweep watchdogs launched =="
echo "   monitor: squeue -u $USER ; tail -f $HERE/output/watchdog_sweep_*.out"
echo "   stop all: pkill -f run_watchdog_finetune.sh"
