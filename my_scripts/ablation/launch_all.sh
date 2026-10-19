#!/bin/bash
# Launch all 17 ablation runs, each under its own auto-requeue watchdog.
# Strategy: submit the baseline first and wait until it has populated the shared
# data_cache (filtered indices + sizes + stats), so the other 16 reuse it instead
# of each re-scanning the dataset.
#
# Usage:  bash launch_all.sh
# Env:    FT_LOGGING (must equal the runs' --logging_path; default below)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${FT_LOGGING:=/ibex/project/c2261/dac_iclr/finetune}"
export FT_LOGGING

echo "== submitting baseline (warms data_cache) =="
bash "$HERE/watchdog_submit.sh" "$HERE/1_baseline.sh"

echo "== waiting for data_cache to warm (sizes.npy under $FT_LOGGING/data_cache; max ~40 min) =="
warm=0
for i in $(seq 1 160); do
    if find "$FT_LOGGING/data_cache" -name 'sizes.npy' 2>/dev/null | grep -q .; then
        echo "   cache warm after ~$(( i * 15 ))s"; warm=1; break
    fi
    sleep 15
done
[ "$warm" -eq 0 ] && echo "   WARN: cache not detected warm; launching anyway (first wave may pay full load)."

echo "== submitting remaining 16 runs =="
for f in 2_gemnet_scratch 3_gemnet_jmp 4_equiformer_jmp \
         5_rbf_gaussian_scratch 6_rbf_bessel_scratch 7_rbf_coulomb_sturmian_scratch \
         8_heteroscedastic 9_classification_head 10_loss_huber 11_loss_l2 \
         12_norm_log 13_uptake_per_cell 14_uptake_per_volume 15_roi_penalty \
         16_invfreq 17_feature_fusion; do
    bash "$HERE/watchdog_submit.sh" "$HERE/${f}.sh"
    sleep 2
done
echo "== all 17 watchdogs launched. Monitor: squeue -u $USER ; tail $HERE/output/watchdog_*.out =="
