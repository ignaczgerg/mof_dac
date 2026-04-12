#!/bin/bash --login
#SBATCH -N 1
#SBATCH --account conf-icl-2025.09.24-ghanembs
#SBATCH --job-name oc20_pretrain
#SBATCH -o output_pretrain/gpu.%A.out
#SBATCH -e output_pretrain/gpu.%A.err
#SBATCH --mail-user=yasir.ghunaim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=170:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=400G
#SBATCH --cpus-per-task=6

#  
#####
hostname
nvidia-smi
conda activate aramco_dac
##### 
#### CUDA_VISIBLE_DEVICES
cd ..
# export OMP_NUM_THREADS=4
# torchrun --nproc_per_node=4 pretrain.py \
#     --lr 4e-4 \
#     --batch_size 16 \
#     --num_workers 6 \
#     --epochs 15 \
#     --model_name "equiformer_v2" \
#     --train_samples_limit 2000000 \
#     --val_samples_limit 2000 \
#     --oc20_split "2M" \
#     --val_interval 5000 \
#     --enable_wandb \
#     --position_norm \
#     --postfix "supervised[single-head]" \
#     --task "oc20" \
#     --resume_from_checkpoint \
#     --load_checkpoint '/ibex/project/c2261/dac_iclr/pretrain/lightning_logs/73nv5u03/aramco_dac_pretrain/73nv5u03/checkpoints/epoch=2-step=87500.ckpt' \
    # --load_checkpoint '/ibex/project/c2261/dac_iclr/pretrain/lightning_logs/6cmw5lh3/aramco_dac_pretrain/6cmw5lh3/checkpoints/epoch=1-step=122500.ckpt' \
    # --autoregressive \
    # --task "oc20,oc22,ani1x,transition1x" \
    # --train_samples_limit 999_000_000 \
    # --no_pbc \
    # --val_interval 0.002986545
    # --max_natoms 600 \
    # --temperature_sampling \
    # --logging_path "testing" \


export OMP_NUM_THREADS=4
torchrun --nproc_per_node=4 --master_port=29789 pretrain.py \
    --lr 4e-4 \
    --batch_size 32 \
    --num_workers 6 \
    --task "oc20" \
    --epochs 30 \
    --model_name "equiformer_v2" \
    --train_samples_limit 2_000_000 \
    --val_samples_limit 2000 \
    --autoregressive \
    --oc20_split "2M" \
    --val_interval 5000 \
    --enable_wandb \
    --enable_mol_shuffle \
    --enable_mol_shuffle_eval \
    --resume_from_checkpoint \
    --load_checkpoint '/ibex/project/c2261/dac_iclr/pretrain/lightning_logs/5zzb58or/aramco_dac_pretrain/5zzb58or/checkpoints/epoch=2-step=82500.ckpt' \
    # --position_norm \
    # --autoregressive \
    # --task "oc20,oc22,ani1x,transition1x" \
    # --position_norm \
    # --enable_wandb \
    # --postfix "supervised[multi-head]" \
    # --resume_from_checkpoint \
    # --load_checkpoint '/ibex/project/c2261/dac_iclr/pretrain/lightning_logs/73nv5u03/aramco_dac_pretrain/73nv5u03/checkpoints/epoch=2-step=87500.ckpt' \
    # --load_checkpoint '/ibex/project/c2261/dac_iclr/pretrain/lightning_logs/6cmw5lh3/aramco_dac_pretrain/6cmw5lh3/checkpoints/epoch=1-step=122500.ckpt' \
    # --train_samples_limit 999_000_000 \
    # --no_pbc \
    # --val_interval 0.002986545
    # --max_natoms 600 \
    # --temperature_sampling \
    # --logging_path "testing" \



# export OMP_NUM_THREADS=8
# torchrun --nproc_per_node=8 pretrain.py \
# python pretrain.py \
#     --lr 4e-4 \
#     --batch_size 2 \
#     --num_workers 6 \
#     --epochs 15 \
#     --model_name "equiformer_v2" \
#     --train_samples_limit 2000000 \
#     --val_samples_limit 2000 \
#     --oc20_split "2M" \
#     --val_interval 5000 \
#     --task "oc20" \
#     --multi_heads \
#     --autoregressive \
#     --resume_from_checkpoint \
#     --load_checkpoint '/ibex/project/c2261/dac_iclr/pretrain/lightning_logs/u2lt6okg/aramco_dac_pretrain/u2lt6okg/checkpoints/epoch=0-step=245000.ckpt' \
#     # --task "oc20,oc22,ani1x,transition1x" \
#     # --position_norm \
#     # --enable_wandb \
#     # --postfix "supervised[multi-head]" \
#     # --load_checkpoint '/ibex/project/c2261/dac_iclr/pretrain/lightning_logs/73nv5u03/aramco_dac_pretrain/73nv5u03/checkpoints/epoch=2-step=87500.ckpt' \
#     # --train_samples_limit 999_000_000 \
#     # --no_pbc \
#     # --val_interval 0.002986545
#     # --max_natoms 600 \
#     # --temperature_sampling \
#     # --logging_path "testing" \
