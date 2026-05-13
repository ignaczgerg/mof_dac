#!/bin/bash --login
#SBATCH -N 1
#SBATCH --account conf-icl-2025.09.24-ghanembs
#SBATCH --job-name odac_pretrain
#SBATCH -o output_pretrain/gpu.%A.out
#SBATCH -e output_pretrain/gpu.%A.err
#SBATCH --mail-user=yasir.ghunaim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
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

export OMP_NUM_THREADS=4
torchrun --nproc_per_node=4 --master_port=29791 pretrain.py \
    --lr 4e-4 \
    --num_ctx_atoms 0.3 \
    --batch_size 8 \
    --num_workers 6 \
    --task "odac" \
    --epochs 30 \
    --model_name "equiformer_v2" \
    --train_samples_limit 2_000_000 \
    --val_samples_limit 2000 \
    --autoregressive \
    --val_interval 5000 \
    --enable_wandb \

