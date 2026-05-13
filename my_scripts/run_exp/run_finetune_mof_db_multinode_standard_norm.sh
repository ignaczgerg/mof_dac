#!/bin/bash --login
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name 6-adsorption-mof-db1
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=350G

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + RANDOM % 50000))

echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Job ID: $SLURM_JOB_ID"

hostname
nvidia-smi
# source $(conda info --base)/etc/profile.d/conda.sh
conda activate aramco_dac

# adsorption_mof_db_anion
# "adsorption_cof_db2"
# "adsorption_db2_merged"
# "adsorption_db1_merged"
# "adsorption_mof_db_1_charges"   
# "adsorption_mof_db_2_charges"   
# "adsorption_mof_db_anion_charges" 

DEBUG_MODE=false
DATASET_NAME="adsorption_aramco"
TASK_TYPE="regression"
MODEL_NAME="equiformer_v2"
TARGETS=("qst_co2" "co2_uptake")
LOSS_COEFFS=(0.5 0.5)
NORMALIZATION_TYPE=("standard" "log")
REDUCTION=("mean" "mean" )
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

export NCCL_TIMEOUT=300
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TORCHELASTIC_ERROR_FILE=./torch_elastic_error_${SLURM_JOB_ID}.json
export TORCH_SHOW_CPP_STACKTRACES=1
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
    --epochs 25 \
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
    --checkpoint_tag "odac_public" \
    --uptake_conversion_mode "per_volume" \
    --heteroscedastic \
    --checkpoint_tag "odac_public" \
    --fusion_feature_names pld lcd gcd unitcell_volume density asa av nav \
    --enable_feature_fusion \
    --fusion_type late \
    $WANDB_FLAG \
    # --use_charge_embedding \
    # --loss "l1" \
