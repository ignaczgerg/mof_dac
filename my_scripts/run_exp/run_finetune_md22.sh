#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name MD22
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=yasir.ghunaim@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6

#####
hostname
nvidia-smi
conda activate aramco_dac
##### 

cd ..
python finetune.py \
    --dataset_name "md22" \
    --targets "DHA" \
    --lr 3.2e-4 \
    --epochs 100 \
    --model_name "equiformer_v2" \
    --small \
    --enable_wandb \
    --checkpoint_tag "oc20_2M_autoreg_epoch_14" \
    # --checkpoint_tag "jmp" \
    # --checkpoint_tag "oc20_public" \
    # --scratch \
    # --checkpoint_tag "oc20_supervised" \
    # --checkpoint_tag "oc20_2M_autoreg_epoch_07" \
    # --checkpoint_path "ani1x_1M" \
    # --targets "Ac-Ala3-NHMe" \