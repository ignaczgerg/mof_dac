#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name MD22
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=48:00:00
### SBATCH --gres=gpu:v100:4
#SBATCH --mem=96G
#SBATCH --cpus-per-task=128

conda activate aramco_dac

python write_lmdb.py \
  --dataset_path /ibex/project/c2261/datasets/hmof \
  --entity_key id \
  --out_dir /ibex/project/c2261/datasets/hmof/lmdb_mini \
  --train_keys_file /ibex/project/c2261/datasets/hmof/split_keys_mini/train_keys.pkl \
  --val_keys_file   /ibex/project/c2261/datasets/hmof/split_keys_mini/val_keys.pkl \
  --test_keys_file  /ibex/project/c2261/datasets/hmof/split_keys_mini/test_keys.pkl \
  --num_workers 64 --backend process --map_size_gb 8 --commit_interval 10000
