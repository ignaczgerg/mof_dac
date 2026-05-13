#!/bin/bash --login
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name oc20_pretrain
#SBATCH -o output_pretrain/gpu.%A.out
#SBATCH -e output_pretrain/gpu.%A.err
#SBATCH --mail-user=yasir.ghunaim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8

hostname
nvidia-smi
conda activate aramco_dac
cd ..

# Get the master node's hostname (first node)
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Default port for rendezvous (optional: randomize to avoid port conflicts)
MASTER_PORT=29610 

# Set rank based on SLURM task ID
NODE_RANK=$SLURM_NODEID

# Auto-detect number of GPUs per node
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
export NPROC_PER_NODE

# Optional: reduce thread contention
export OMP_NUM_THREADS=4

ping -c 3 $MASTER_ADDR

echo "==== SLURM Torch Distributed Job ===="
echo "MASTER_ADDR     = $MASTER_ADDR"
echo "MASTER_PORT     = $MASTER_PORT"
echo "NODE_RANK       = $NODE_RANK"
echo "SLURM_JOB_NUM_NODES = $SLURM_JOB_NUM_NODES"
echo "WORLD_SIZE      = $((NPROC_PER_NODE * SLURM_JOB_NUM_NODES))"
echo "NPROC_PER_NODE  = $NPROC_PER_NODE"
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "======================================"

# Run torchrun
torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --node_rank=$NODE_RANK \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  pretrain.py \
    --lr 4e-4 \
    --batch_size 16 \
    --num_workers 6 \
    --task "oc20" \
    --epochs 30 \
    --model_name "equiformer_v2" \
    --train_samples_limit 2000000 \
    --val_samples_limit 2000 \
    --oc20_split "2M" \
    --val_interval 5000 \
    --enable_wandb \
    --autoregressive
    # --position_norm \

# export OMP_NUM_THREADS=4
# torchrun --nproc_per_node=4 pretrain.py \
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
#     # --autoregressive \
#     # --task "oc20,oc22,ani1x,transition1x" \
#     # --position_norm \
#     # --enable_wandb \
#     # --postfix "supervised[multi-head]" \
#     # --resume_from_checkpoint \
#     # --load_checkpoint '/ibex/project/c2261/dac_iclr/pretrain/lightning_logs/73nv5u03/aramco_dac_pretrain/73nv5u03/checkpoints/epoch=2-step=87500.ckpt' \
#     # --load_checkpoint '/ibex/project/c2261/dac_iclr/pretrain/lightning_logs/6cmw5lh3/aramco_dac_pretrain/6cmw5lh3/checkpoints/epoch=1-step=122500.ckpt' \
#     # --train_samples_limit 999_000_000 \
#     # --no_pbc \
#     # --val_interval 0.002986545
#     # --max_natoms 600 \
#     # --temperature_sampling \
#     # --logging_path "testing" \