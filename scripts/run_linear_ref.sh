#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name=gen-metadata
#SBATCH -o output/cpu.%A.out
#SBATCH -e output/cpu.%A.err
#SBATCH --mail-user=yasir.ghunaim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=2:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=24

# === Load environment ===
hostname
conda activate aramco_dac

cd ..
# === Run metadata script ===
python -m jmp.datasets.scripts.odac_preprocess.odac_linear_ref linref \
--src /ibex/project/c2261/datasets/odac/s2ef/val \
--out_path /ibex/project/c2261/datasets/odac/s2ef/linref.npz \
--num_workers 24

python -m jmp.datasets.scripts.odac_preprocess.odac_linear_ref compute_mean_std \
--src /ibex/project/c2261/datasets/odac/s2ef/val \
--linref_path /ibex/project/c2261/datasets/odac/s2ef/linref.npz \
--out_path /ibex/project/c2261/datasets/odac/s2ef/mean_std.pkl \
--num_workers 100

