#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name charge
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

hostname
nvidia-smi
conda activate aramco_dac

DEBUG_MODE=False
# the available datasets:
# adsorption_mof_db_anion
# "adsorption_cof_db2"
# "adsorption_db2_merged"
# "adsorption_db1_merged"
# "adsorption_mof_db_1_charges"   
# "adsorption_mof_db_2_charges"   
# "adsorption_mof_db_anion_charges" 
DATASET_NAME="adsorption_aramco"
TASK_TYPE="regression"
MODEL_NAME="equiformer_v2"

TARGETS=("qst_co2" "co2_uptake")
LOSS_COEFFS=(0.5 0.5)
NORMALIZATION_TYPE=("standard" "log")
REDUCTION=("mean" "mean")

WANDB_FLAG="--enable_wandb"

if [ "$DEBUG_MODE" = true ]; then
    WANDB_FLAG=""
fi

NUM_DISTANCE_BASIS_LIST=(600)
RBF_LIST=("gaussian")

task_id=${SLURM_ARRAY_TASK_ID:-0}
num_rbf=${#RBF_LIST[@]}

basis_idx=$(( task_id / num_rbf ))
rbf_idx=$(( task_id % num_rbf ))

NUM_DISTANCE_BASIS=${NUM_DISTANCE_BASIS_LIST[$basis_idx]}
RBF_FUNCTION=${RBF_LIST[$rbf_idx]}

echo "SLURM_ARRAY_TASK_ID = $task_id"
echo "Selected num_distance_basis = $NUM_DISTANCE_BASIS"
echo "Selected rbf_function = $RBF_FUNCTION"

cd ..

# torchrun --nproc_per_node=4 --master_port=$((10000 + RANDOM % 50000)) 
python finetune.py \
    --tasks "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --lr 1.0e-4 \
    --epochs 60 \
    --num_workers 4 \
    --weight_decay 1e-3 \
    --normalization_type "${NORMALIZATION_TYPE[@]}" \
    --graph_scalar_reduction "${REDUCTION[@]}" \
    --small \
    --atom_bucket_batch_sampler \
    --max_atoms_per_batch 800 \
    --max_natoms 800 \
    --cutoff 6 \
    --rbf_function "$RBF_FUNCTION" \
    --num_distance_basis "$NUM_DISTANCE_BASIS" \
    --max_neighbors 32 \
    --val_same_as_train \
    --uptake_conversion_mode "per_volume" \
    --loss "l2" \
    --train_samples_limit 0.05 \
    --test_samples_limit 100 \
    --val_samples_limit 100 \
    --targets "${TARGETS[@]}" \
    --heteroscedastic \
    --checkpoint_tag "odac_public" \
    --fusion_feature_names pld lcd gcd unitcell_volume density asa av nav \
    --enable_feature_fusion \
    --fusion_type late \
    --fusion_hidden_dim 128 \
    # --use_charge_embedding \
    # $WANDB_FLAG \
    # --scratch \
    # --targets "${TARGETS[@]}" \
    # --classification_targets co2_uptake \
    # --classification_threshold 0.5 \
    # --classification_loss_weight 1.0 \
