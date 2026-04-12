#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name QMOF
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=hani.majed@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-2   # 3 runs: index 0, 1, 2, 3, 4, 5



# List of checkpoint tags
CHECKPOINT_TAGS_NONPOS=("all_datasets_8M_autoreg_singleHead_epoch_09" "all_datasets_2M_autoreg_singleHead_epoch_29" "all_datasets_2M_autoreg_TmpSampling_singleHead_epoch_29")
CHECKPOINT_TAGS=("all_datasets_8M_autoreg_posNorm_singleHead_epoch_09" "all_datasets_2M_autoreg_posNorm_singleHead_epoch_29" "all_datasets_2M_autoreg_TmpSampling_posNorm_singleHead_epoch_29")

### #SBATCH --array=0-2   # 3 runs: index 0, 1, 2

#####
hostname
nvidia-smi
conda activate aramco_dac
##### 

cd ..
python finetune.py \
    --dataset_name "qmof" \
    --batch_size 16 \
    --targets "y" \
    --lr 3.2e-4 \
    --epochs 100 \
    --model_name "equiformer_v2" \
    --small \
    --weight_decay 0.1 \
    --dropout 0.0 \
    --enable_wandb \
    --checkpoint_tag "${CHECKPOINT_TAGS[$SLURM_ARRAY_TASK_ID]}" \
    --position_norm \
    # --scratch \
    # --checkpoint_tag "all_datasets_2M_autoreg_singleHead_epoch_29" \
    # --checkpoint_tag "all_datasets_2M_autoreg_singleHead_epoch_29" \
    # --position_norm \
    # --scratch \
    # --checkpoint_tag "all_datasets_8M_autoreg_epoch_02" \
    # --position_norm \
    # --lr 3.2e-4 \
    # --scratch \
    # --checkpoint_tag "oc20_public" \
    # --checkpoint_tag "oc20_supervised" \
    # --checkpoint_tag "oc20_2M_autoreg_epoch_14" \
    # --checkpoint_tag "oc20_2M_autoreg_epoch_11" \
    # --checkpoint_tag "oc20_2M_autoreg_epoch_03" \
    # --checkpoint_tag "oc20_autoreg_old" \
    # --checkpoint_tag "jmp" \
