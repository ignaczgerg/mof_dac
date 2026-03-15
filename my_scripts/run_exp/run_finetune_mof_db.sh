#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name flowm
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=90:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16

hostname
nvidia-smi
conda activate aramco_dac

DEBUG_MODE=True

# the available datasets:
# "adsorption_cof_db2"
# "adsorption_db2_merged"
DATASET_NAME="adsorption_db1_merged"
TASK_TYPE="regression"
MODEL_NAME="equiformer_v2"

TARGETS=("qst_co2" "kh_co2" "co2_uptake")
LOSS_COEFFS=(0.4 0.3 0.3)
NORMALIZATION_TYPE=("standard" "log" "log")
REDUCTION=("mean" "mean" "mean")

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
    --targets "${TARGETS[@]}" \
    --model_name "$MODEL_NAME" \
    --lr 8e-5 \
    --epochs 25 \
    --num_workers 4 \
    --weight_decay 5e-3 \
    --normalization_type "${NORMALIZATION_TYPE[@]}" \
    --graph_scalar_reduction "${REDUCTION[@]}" \
    --small \
    --atom_bucket_batch_sampler \
    --max_atoms_per_batch 800 \
    --max_natoms 800 \
    --cutoff 6 \
    --checkpoint_tag "odac_public" \
    --rbf_function "$RBF_FUNCTION" \
    --num_distance_basis "$NUM_DISTANCE_BASIS" \
    --max_neighbors 32 \
    --val_same_as_train \
    --heteroscedastic \
    --asymmetric_regression \
    --asymmetric_tau 0.9 \
    --enable_classification \
    --classification_targets co2_uptake \
    --classification_threshold 0.5 \
    --classification_loss_weight 1.0 \
    --focal_gamma 2.0 \
    --focal_alpha_pos 0.9 \
    --focal_alpha_neg 0.1 \
    --invfreq \
    --invfreq-beta 0.75 \
    --val_samples_limit 100 \
    --train_samples_limit 100 \
    --test_samples_limit 100 \
    # $WANDB_FLAG \