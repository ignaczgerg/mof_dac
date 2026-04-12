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


# === Run metadata script ===
# python generate_metadata.py \
#     --src /ibex/ai/reference/OPC_OpenCatalystProject/data/is2re/all/train/ \
#     --dest /ibex/project/c2261/datasets/oc20_is2re/all/train_metadata.npz \
#     --type pretrain \
#     --num-workers 32


python generate_metadata.py \
    --src /ibex/ai/reference/OPC_OpenCatalystProject/data/is2re/all/val_id/ \
    --dest /ibex/project/c2261/datasets/oc20_is2re/all/val_id_metadata.npz \
    --type pretrain \
    --num-workers 32