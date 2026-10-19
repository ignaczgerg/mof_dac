#!/bin/bash --login
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=350G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + RANDOM % 50000))
conda activate aramco_dac
cd ..

RUN_TAG="ablation_A2_gemnet_jmp_v2"

srun --cpu-bind=none python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv-conf timeout=300 \
    finetune.py \
    --tasks adsorption_aramco \
    --targets qst_co2 co2_uptake \
    --targets_loss_coefficients 0.5 0.5 \
    --model_name gemnet \
    --small \
    --loss l1 \
    --normalization_type standard standard \
    --graph_scalar_reduction mean mean \
    --uptake_conversion_mode none \
    --roi_penalty 1.0 \
    --lr 1.0e-4 --epochs 25 --weight_decay 5e-3 --seed 0 --num_workers 4 \
    --atom_bucket_batch_sampler --max_atoms_per_batch 700 --max_natoms 700 \
    --cutoff 6 --max_neighbors 30 \
    --val_same_as_train \
    --run_tag "$RUN_TAG" --save_every_n_train_steps 500 \
    --wandb_project aramco_ablation \
    --enable_wandb
