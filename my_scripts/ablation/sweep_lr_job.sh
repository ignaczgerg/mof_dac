#!/bin/bash --login
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=350G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH -o output/sweep.%A.out
#SBATCH -e output/sweep.%A.err

# LR proxy run for ONE (regime, lr) pair. Driven by sweep_watchdog_submit.sh
# (via run_watchdog_finetune.sh); not meant to be sbatch'd bare.
#
# Env in:  SWEEP_REGIME  one of eqv2_pt | eqv2_scratch | gemnet_pt | gemnet_scratch
#          SWEEP_LR      learning rate, e.g. 3.0e-4
#          SWEEP_EPOCHS  proxy schedule length   (default 15)
#          SWEEP_FRAC    train subset fraction   (default 0.35)
#          SAVE_EVERY    checkpoint interval     (default 500; set by the watchdog)
#
# NOTE: no `set -eu` here. conda's MKL activation hook dereferences an unbound
# MKL_INTERFACE_LAYER, so `set -u` kills the job at `conda activate` (4s FAILED).
# The ablation scripts omit it for the same reason; guards below are explicit.

: "${SWEEP_EPOCHS:=15}"
: "${SWEEP_FRAC:=0.35}"
: "${SAVE_EVERY:=500}"

if [ -z "${SWEEP_REGIME:-}" ] || [ -z "${SWEEP_LR:-}" ]; then
    echo "ERROR: SWEEP_REGIME and SWEEP_LR must be set (submit via sweep_watchdog_submit.sh)" >&2
    exit 2
fi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
conda activate aramco_dac
cd ..

# Regime -> backbone/pretraining flags. These mirror the ablation runs exactly:
#   eqv2_pt        = A4 baseline regime (also covers A3 and all of blocks C-I)
#   eqv2_scratch   = block B regime (B1 gaussian reference)
#   gemnet_pt      = A2   |   gemnet_scratch = A1
# GemNet must NOT receive --rbf_function / --num_distance_basis: GOCBackboneConfig
# has no such fields and finetune.py assigns them onto the frozen config.
#
# WARMUP is set per regime to preserve each regime's real warmup FRACTION at the
# shortened horizon: EquiformerV2 runs ceil(0.01*25)=1 of 25 epochs (4%), GemNet
# runs 5 of 25 (20%). Sweeping both at the same warmup would change the schedule
# shape relative to the runs the LR is being chosen for.
case "$SWEEP_REGIME" in
    eqv2_pt)
        ARCH_FLAGS=(--model_name equiformer_v2 --checkpoint_tag odac_public
                    --rbf_function gaussian --num_distance_basis 600)
        WARMUP=1 ;;
    eqv2_scratch)
        ARCH_FLAGS=(--model_name equiformer_v2 --scratch
                    --rbf_function gaussian --num_distance_basis 600)
        WARMUP=1 ;;
    gemnet_pt)
        ARCH_FLAGS=(--model_name gemnet)
        WARMUP=3 ;;
    gemnet_scratch)
        ARCH_FLAGS=(--model_name gemnet --scratch)
        WARMUP=3 ;;
    *)
        echo "ERROR: unknown SWEEP_REGIME='$SWEEP_REGIME'" >&2; exit 2 ;;
esac

RUN_TAG="sweep_${SWEEP_REGIME}_lr${SWEEP_LR}"

srun --cpu-bind=none python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv-conf timeout=300 \
    finetune.py \
    --tasks adsorption_aramco \
    --targets qst_co2 co2_uptake \
    --targets_loss_coefficients 0.5 0.5 \
    "${ARCH_FLAGS[@]}" \
    --small \
    --loss l1 \
    --normalization_type standard standard \
    --graph_scalar_reduction mean mean \
    --uptake_conversion_mode none \
    --roi_penalty 1.0 \
    --lr "$SWEEP_LR" --epochs "$SWEEP_EPOCHS" --warmup_epochs "$WARMUP" \
    --weight_decay 5e-3 --seed 0 --num_workers 4 \
    --train_samples_limit "$SWEEP_FRAC" \
    --atom_bucket_batch_sampler --max_atoms_per_batch 700 --max_natoms 700 \
    --cutoff 6 --max_neighbors 30 \
    --val_same_as_train \
    --run_tag "$RUN_TAG" --save_every_n_train_steps "$SAVE_EVERY" \
    --wandb_project aramco_lr_sweep \
    --enable_wandb
