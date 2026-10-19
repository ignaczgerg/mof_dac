#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name=moftr-f2
#SBATCH -o output/f2.%A_%a.out
#SBATCH -e output/f2.%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda activate aramco_dac
cd "$SLURM_SUBMIT_DIR"

: "${CAMPAIGN_DIR:=/ibex/project/c2261/dac_iclr/predictions/mof_transformer_ablation_F2}"
: "${CODE_DIR:=$CAMPAIGN_DIR/code_d89554d/src}"
: "${CKPT:=/ibex/project/c2261/dac_iclr/finetune/lightning_logs/ablation_F2_per_cell_v2/aramco_ablation/ablation_F2_per_cell_v2/checkpoints/last.ckpt}"
: "${SHARD_SIZE:=7500}"
: "${MAX_NATOMS:=700}"
: "${MAX_ATOMS_PER_BATCH:=700}"
: "${OUT_DIR:=$CAMPAIGN_DIR/shards}"
: "${LIMIT:=}"
: "${VALIDATE_LMDB:=}"

export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

hostname
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ -n "$VALIDATE_LMDB" ]; then
    python predict_shard_ablation.py --ckpt "$CKPT" --max_natoms "$MAX_NATOMS" \
        --max_atoms_per_batch "$MAX_ATOMS_PER_BATCH" --num_workers 8 \
        --validate_lmdb "$VALIDATE_LMDB" \
        --reference_dir /ibex/project/c2261/dac_iclr/finetune/ablation_eval/ablation_F2_per_cell_v2/predictions/final/test/adsorption_aramco
    exit $?
fi

EXTRA=""
if [ -n "$LIMIT" ]; then
    EXTRA="--limit $LIMIT"
fi

python predict_shard_ablation.py \
    --ckpt "$CKPT" \
    --manifest "$CAMPAIGN_DIR/manifest.txt" \
    --out "$OUT_DIR" \
    --shard_id "$SLURM_ARRAY_TASK_ID" \
    --shard_size "$SHARD_SIZE" \
    --max_natoms "$MAX_NATOMS" \
    --max_atoms_per_batch "$MAX_ATOMS_PER_BATCH" \
    --num_workers 8 \
    --prefilter_workers 8 \
    $EXTRA
