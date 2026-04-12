#!/bin/bash --login
#SBATCH -N 1
#SBATCH --account conf-icl-2025.09.24-ghanembs
#SBATCH --job-name all_pretrain
#SBATCH -o output_pretrain/gpu.%A.out
#SBATCH -e output_pretrain/gpu.%A.err
#SBATCH --mail-user=yasir.ghunaim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=170:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=600G
#SBATCH --cpus-per-task=10

#  
#####
hostname
nvidia-smi
conda activate aramco_dac
##### 
#### CUDA_VISIBLE_DEVICES
cd ..
## For A100
export OMP_NUM_THREADS=4
torchrun --nproc_per_node=4 --master_port=29800 pretrain.py \
    --lr 4e-4 \
    --batch_size 16 \
    --num_workers 6 \
    --task "oc20,oc22,ani1x,transition1x" \
    --epochs 30 \
    --model_name "equiformer_v2" \
    --train_samples_limit 2_000_000 \
    --val_samples_limit 2000 \
    --autoregressive \
    --val_interval 5000 \
    --enable_wandb \
    --enable_mol_shuffle \
    --enable_mol_shuffle_eval \
    --position_norm \
    # --train_samples_limit 2_000_000 \
    # --temperature_sampling \
    # --position_norm \
    # --train_samples_limit 999_000_000 \
    # --no_pbc
    # --enable_wandb \
    # --val_interval 0.002986545
    # --max_natoms 600 \
    # --temperature_sampling \
    # --task "oc20,oc22,ani1x,transition1x" \

# ### For debugging
# python pretrain.py \
#     --lr 4e-4 \
#     --batch_size 16 \
#     --num_workers 6 \
#     --task "oc20,oc22,ani1x,transition1x" \
#     --epochs 30 \
#     --model_name "equiformer_v2" \
#     --train_samples_limit 80_000 \
#     --val_samples_limit 2000 \
#     --autoregressive \
#     --val_interval 5000 \
#     --position_norm \
#     # --train_samples_limit 2_000_000 \
#     # --temperature_sampling \
#     # --position_norm \
#     # --train_samples_limit 999_000_000 \
#     # --no_pbc
#     # --enable_wandb \
#     # --val_interval 0.002986545
#     # --max_natoms 600 \
#     # --temperature_sampling \
#     # --task "oc20,oc22,ani1x,transition1x" \

# # For v100
# export OMP_NUM_THREADS=4
# torchrun --nproc_per_node=4 pretrain.py \
#     --lr 4e-4 \
#     --batch_size 5 \
#     --num_workers 6 \
#     --task "oc20" \
#     --epochs 1 \
#     --model_name "equiformer_v2" \
#     --val_samples_limit 2000 \
#     --autoregressive \
#     --oc20_split "all" \
#     --val_interval 5000 \
#     --enable_wandb \
#     # --train_samples_limit 999_000_000 \
#     # --no_pbc
#     # --enable_wandb \
#     # --val_interval 0.002986545
#     # --max_natoms 600 \
#     # --temperature_sampling \
#     # --task "oc20,oc22,ani1x,transition1x" \

