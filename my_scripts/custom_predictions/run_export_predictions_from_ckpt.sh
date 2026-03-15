#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name inference
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1


hostname
nvidia-smi
conda activate aramco_dac

python export_predictions_from_ckpt.py \
  --ckpt /ibex/project/c2261/dac_iclr/finetune/lightning_logs/yozti0u8/aramco_dac_finetune/yozti0u8/checkpoints/last.ckpt \
  --device cuda \
  --num_workers 4 \
  # --train-as-test