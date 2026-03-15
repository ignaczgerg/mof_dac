#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name adsorption-mof-db1
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8


hostname
nvidia-smi
conda activate aramco_dac

python custom_dataset_evaluate.py \
  --ckpt /ibex/project/c2261/dac_iclr/finetune/lightning_logs/yozti0u8/aramco_dac_finetune/yozti0u8/checkpoints/last.ckpt \
  --task_name external_eval \
  --test_src /ibex/project/c2261/datasets/adsorption-mof-db1/lmdb/test \
  --predict_only \
  --suffix "_db1" \
  --num_workers 6 \
  --denorm_task adsorption_db_1 \

# python custom_dataset_evaluate.py \
#   --ckpt /ibex/project/c2261/dac_iclr/finetune/lightning_logs/yozti0u8/aramco_dac_finetune/yozti0u8/checkpoints/last.ckpt \
#   --task_name external_eval \
#   --test_src /ibex/project/c2261/datasets/adsorption-mof-ai-generated-batch-3/lmdb/test \
#   --predict_only \
#   --suffix "batch3" \
#   --num_workers 6 \
#   --denorm_task adsorption_db_2 \


# python custom_dataset_evaluate.py \
#   --ckpt /ibex/project/c2261/dac_iclr/finetune/lightning_logs/yozti0u8/aramco_dac_finetune/yozti0u8/checkpoints/last.ckpt \
#   --task_name external_eval \
#   --test_src /ibex/project/c2261/datasets/adsorption-mof-ai-generated-batch-1/lmdb/test \
#   --predict_only \
#   --suffix "batch1" \
#   --num_workers 6 \
#   --denorm_task adsorption_db_2 \


# python custom_dataset_evaluate.py \
#   --ckpt /ibex/project/c2261/dac_iclr/finetune/lightning_logs/yozti0u8/aramco_dac_finetune/yozti0u8/checkpoints/last.ckpt \
#   --task_name external_eval \
#   --test_src /ibex/project/c2261/datasets/adsorption-mof-db2/lmdb/test \
#   --predict_only \
#   --suffix "db2" \
#   --num_workers 6 \

