#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name hmof
#SBATCH -o output/gpu.%A.%a.out
#SBATCH -e output/gpu.%A.%a.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-11
#SBATCH --account conf-icl-2025.09.24-ghanembs

#####
hostname
nvidia-smi
conda activate aramco_dac
#####

cd ..

CHECKPOINT_TAGS_ALL=( 
    "scratch"
    "scratch"
    "oc20_autoreg_single_vector_head"
    "oc20_autoreg_single_vector_head"
    "oc20_permutation_invariant"
    "oc20_permutation_invariant"
    "oc20_public"
    "oc20_public"
    "odac_public"
    "odac_public"
    "odac_autoreg_permutation_invariant"
    "odac_autoreg_permutation_invariant"
)
    # "all_datasets_2M_autoreg_TmpSampling_singleHead_epoch_29"
    # "all_datasets_2M_autoreg_singleHead_epoch_29"

# LRS=( "1e-6" "1e-6" "1e-6" "3e-5" "3e-5" "1e-5" )
# WDS=( "0.0"  "1e-5" "1e-4" "1e-5" "1e-4" "1e-3" )

# LR=${LRS[$SLURM_ARRAY_TASK_ID]}
# WD=${WDS[$SLURM_ARRAY_TASK_ID]}

if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then
    TARGET="qst_binary_n2"
    NORMALIZATION_TYPE="standard"
else
    TARGET="uptake_binary_n2"
    NORMALIZATION_TYPE="log"
fi

if [[ "${CHECKPOINT_TAGS_ALL[$SLURM_ARRAY_TASK_ID]}" == "scratch" ]]; then
    CHECKPOINT_OPTION="--scratch"
else
    CHECKPOINT_OPTION=( --checkpoint_tag "${CHECKPOINT_TAGS_ALL[$SLURM_ARRAY_TASK_ID]}" )
fi

# CHECKPOINT_TAG=${CHECKPOINT_TAGS_ALL[$SLURM_ARRAY_TASK_ID]}
MAX_ATOMS_PER_BATCH=2000
python finetune.py \
    --dataset_name "hmof-mini" \
    --targets $TARGET \
    --normalization_type $NORMALIZATION_TYPE \
    --graph_scalar_reduction "mean" \
    --atom_bucket_batch_sampler \
    --num_workers 4 \
    --batch_size 24 \
    --lr 8e-5 \
    --epochs 60 \
    --model_name "equiformer_v2" \
    --small \
    --weight_decay 5e-3 \
    --dropout 0.0 \
    --enable_wandb \
   "${CHECKPOINT_OPTION[@]}"
    # --checkpoint_tag "${CHECKPOINT_TAGS_ALL[$SLURM_ARRAY_TASK_ID]}" \
    # --checkpoint_tag "oc20_public" \
    # --scratch \
    # $POSITION_NORM