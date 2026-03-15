#!/bin/bash --login
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name 6-adsorption-mof-db1
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=350G

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + RANDOM % 50000))

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Job ID: $SLURM_JOB_ID"

hostname
nvidia-smi
conda activate aramco_dac

DEBUG_MODE=false
DATASET_NAME="adsorption_db1_merged adsorption_db2_merged"
TASK_TYPE="regression"
MODEL_NAME="equiformer_v2"
TARGETS=("qst_co2" "kh_co2" "co2_uptake")
LOSS_COEFFS=(0.4 0.3 0.3)
NORMALIZATION_TYPE=("standard" "log" "log")
REDUCTION=("mean" "mean" "mean")
WANDB_FLAG="--enable_wandb"
RECIPROCAL_HEAD=""

if [ "$DEBUG_MODE" = true ]; then WANDB_FLAG=""; fi
if [ "$NO_RECIPROCAL" = true ]; then RECIPROCAL_HEAD="--no_reciprocal_block"; fi

NUM_DISTANCE_BASIS_LIST=(600)
RBF_LIST=("gaussian")

task_id=${SLURM_ARRAY_TASK_ID:-0}
num_rbf=${#RBF_LIST[@]}
basis_idx=$(( task_id / num_rbf ))
rbf_idx=$(( task_id % num_rbf ))
NUM_DISTANCE_BASIS=${NUM_DISTANCE_BASIS_LIST[$basis_idx]}
RBF_FUNCTION=${RBF_LIST[$rbf_idx]}

echo "SLURM_ARRAY_TASK_ID = $task_id"
cd ..

srun --cpu-bind=none torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    finetune.py \
    --tasks "$DATASET_NAME" \
    --targets "${TARGETS[@]}" \
    --model_name "$MODEL_NAME" \
    --lr 3.2e-4 \
    --epochs 25 \
    --num_workers 4 \
    --weight_decay 5e-3 \
    --normalization_type "${NORMALIZATION_TYPE[@]}" \
    --graph_scalar_reduction "${REDUCTION[@]}" \
    --small \
    $WANDB_FLAG \
    --atom_bucket_batch_sampler \
    --max_atoms_per_batch 800 \
    --max_natoms 800 \
    --cutoff 6 \
    --rbf_function "$RBF_FUNCTION" \
    --num_distance_basis "$NUM_DISTANCE_BASIS" \
    --max_neighbors 32 \
    --checkpoint_tag "odac_public" \
    --val_same_as_train \
    --heteroscedastic \
    --asymmetric_regression \
    --asymmetric_tau 0.9 \
    --enable_classification \
    --classification_targets co2_uptake \
    --classification_threshold 0.6 \
    --classification_loss_weight 1.0 \
    --focal_gamma 2.0 \
    --focal_alpha_pos 0.9 \
    --focal_alpha_neg 0.1 \
    --invfreq \
    --invfreq-beta 0.75 \
