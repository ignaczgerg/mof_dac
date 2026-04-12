#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name=QM9
#SBATCH -o output/gpu.%A.%a.out
#SBATCH -e output/gpu.%A.%a.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=6
###SBATCH --array=0-3

nvidia-smi
conda activate aramco_dac
cd ..
# CHECKPOINT_TAGS_ALL=( 
#     "oc20_autoreg_single_vector_head"
#     "oc20_permutation_invariant"
#     "all_datasets_2M_autoreg_TmpSampling_singleHead_epoch_29"
#     "all_datasets_2M_autoreg_singleHead_epoch_29"
# )

# CHECKPOINT_TAG=${CHECKPOINT_TAGS_ALL[$SLURM_ARRAY_TASK_ID]}

# torchrun --nproc_per_node=4 --master_port=$((10000 + RANDOM % 50000))
python finetune.py \
    --batch_size 128 \
    --dataset_name qm9 \
    --graph_scalar_reduction sum \
    --targets mu \
    --lr 3e-4 \
    --epochs 300 \
    --model_name equiformer_v2 \
    --small \
    --weight_decay 0.0 \
    --dropout 0.0 \
    --enable_wandb \
    --scratch \
    # --checkpoint_tag "$CHECKPOINT_TAG" \
# --checkpoint_tag oc20_autoreg_single_vector_head \ 