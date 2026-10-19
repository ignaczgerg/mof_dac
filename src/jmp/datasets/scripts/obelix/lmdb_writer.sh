#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name MD22
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=0:30:00
### SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

conda activate aramco_dac

python write_lmdb.py \
  --dataset_path /ibex/project/c2261/datasets/obelix \
  --entity_key ID \
  --out_dir /ibex/project/c2261/datasets/obelix/lmdb \
  --num_workers 4 --map_size_gb 1 --commit_interval 10 --log_level INFO
