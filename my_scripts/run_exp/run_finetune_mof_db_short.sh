#!/bin/bash --login
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o output/short.%A.out
#SBATCH -e output/short.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=350G
if [ -z "${RUN_TAG:-}" ]; then
    echo "RUN_TAG is required (export it before sbatch, or use the watchdog)." >&2
    exit 2
fi
: "${SAVE_EVERY:=200}"
: "${MAX_EPOCHS:=25}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + RANDOM % 50000))

echo "RUN_TAG=$RUN_TAG  SAVE_EVERY=$SAVE_EVERY  MAX_EPOCHS=$MAX_EPOCHS"
echo "Master node: $MASTER_ADDR  Master port: $MASTER_PORT"
echo "Job ID: $SLURM_JOB_ID"

hostname
nvidia-smi
# set +u
conda activate aramco_dac
# set -u

DEBUG_MODE=false
DATASET_NAME="adsorption_aramco"
MODEL_NAME="equiformer_v2"
TARGETS=("qst_co2" "co2_uptake")
NORMALIZATION_TYPE=("standard" "standard")
REDUCTION=("sum" "sum")
WANDB_FLAG="--enable_wandb"

if [ "$DEBUG_MODE" = true ]; then WANDB_FLAG=""; fi

NUM_DISTANCE_BASIS=600
RBF_FUNCTION="gaussian"

cd ..

export NCCL_TIMEOUT=300
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TORCHELASTIC_ERROR_FILE=./torch_elastic_error_${SLURM_JOB_ID}.json
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONFAULTHANDLER=1

srun --cpu-bind=none python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv-conf timeout=300 \
    finetune.py \
    --tasks "$DATASET_NAME" \
    --targets "${TARGETS[@]}" \
    --model_name "$MODEL_NAME" \
    --lr 1.0e-4 \
    --epochs "$MAX_EPOCHS" \
    --num_workers 4 \
    --weight_decay 5e-3 \
    --normalization_type "${NORMALIZATION_TYPE[@]}" \
    --graph_scalar_reduction "${REDUCTION[@]}" \
    --small \
    --atom_bucket_batch_sampler \
    --max_atoms_per_batch 700 \
    --max_natoms 700 \
    --cutoff 6 \
    --rbf_function "$RBF_FUNCTION" \
    --num_distance_basis "$NUM_DISTANCE_BASIS" \
    --max_neighbors 30 \
    --val_same_as_train \
    --loss "l1" \
    --run_tag "$RUN_TAG" \
    --save_every_n_train_steps "$SAVE_EVERY" \
    --scratch \
    $WANDB_FLAG
# --checkpoint_tag "odac_public" \
