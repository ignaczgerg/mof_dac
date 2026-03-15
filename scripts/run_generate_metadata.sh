#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name=gen-metadata
#SBATCH -o output/cpu.%A.out
#SBATCH -e output/cpu.%A.err
#SBATCH --mail-user=yasir.ghunaim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=20:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=100

# === Load environment ===
hostname
echo "Starting metadata generation..."
conda activate aramco_dac


=== Run metadata script ===
# python generate_metadata.py \
#     --src /ibex/project/c2261/datasets/transition1x/lmdb/train/ \
#     --dest ./train_transition1x_metadata.npz \
#     --type pretrain \
#     --num-workers 60

python generate_metadata.py \
    --src /ibex/ai/reference/OPC_OpenCatalystProject/data/oc22/s2ef-total/train/ \
    --dest ./train_oc22_2M_metadata.npz \
    --type pretrain \
    --num-workers 60

