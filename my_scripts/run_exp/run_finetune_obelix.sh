#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name hmof
#SBATCH -o output/gpu.%A.%a.out
#SBATCH -e output/gpu.%A.%a.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-5
#SBATCH --account conf-icl-2025.09.24-ghanembs

#####
hostname
nvidia-smi
conda activate aramco_dac
#####

cd ..

CHECKPOINT_TAGS_ALL=( 
    "scratch"
    "oc20_autoreg_single_vector_head"
    "oc20_permutation_invariant"
    "oc20_public"
    "odac_public"
    "odac_autoreg_permutation_invariant"
)


if [[ "${CHECKPOINT_TAGS_ALL[$SLURM_ARRAY_TASK_ID]}" == "scratch" ]]; then
    CHECKPOINT_OPTION="--scratch"
else
    CHECKPOINT_OPTION=( --checkpoint_tag "${CHECKPOINT_TAGS_ALL[$SLURM_ARRAY_TASK_ID]}" )
fi

MAX_ATOMS_PER_BATCH=2000
python finetune.py \
    --dataset_name "obelix" \
    --targets "ionic_conductivity" \
    --normalization_type "log" \
    --graph_scalar_reduction "mean" \
    --atom_bucket_batch_sampler \
    --num_workers 4 \
    --batch_size 4 \
    --lr 8e-5 \
    --epochs 15 \
    --model_name "equiformer_v2" \
    --small \
    --weight_decay 5e-3 \
    --dropout 0.0 \
    --enable_wandb \
   "${CHECKPOINT_OPTION[@]}"
    # --checkpoint_tag "${CHECKPOINT_TAGS_ALL[$SLURM_ARRAY_TASK_ID]}" \
    # --checkpoint_tag "oc20_public" \
    # $POSITION_NORM