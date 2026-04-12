#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name MD17
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32

#####
hostname
nvidia-smi
conda activate aramco_dac
##### 

cd ..
# export OMP_NUM_THREADS=4
MASTER_PORT=$((12000 + RANDOM % 10000))
# torchrun --nproc_per_node=4 --master_port=$MASTER_PORT
python finetune.py \
    --dataset_name "rmd17" \
    --targets "aspirin" \
    --batch_size 8 \
    --model_name "equiformer_v2" \
    --num_workers 4 \
    --small \
    --lr 5e-4 \
    --weight_decay 1e-6 \
    --epochs 1500 \
    --scratch \
    --compute_avg_dataset_stats \
    --compute_avg_dataset_stats_degree \
    # --enable_wandb \
    # --position_norm \
    # --checkpoint_tag "all_datasets_8M_autoreg_posNorm_singleHead_epoch_08" \


    # --checkpoint_tag "jmp_autoreg_epoch_14" \
    # --checkpoint_tag "oc20_public" \
    # --checkpoint_tag "oc20_supervised" \
    # --checkpoint_tag "ani1x_2M_supervised_epoch_14" \
    # --checkpoint_tag "ani1x_2M_autoreg_epoch_14" \
    # --checkpoint_tag "oc20_2M_autoreg_epoch_14" \
    # --checkpoint_tag "oc20_public" \
    # --scratch \
    
    

    # --checkpoint_tag "jmp" \
    # --checkpoint_tag "oc20_2M_autoreg_epoch_03" \
    
    
    # --checkpoint_tag "jmp_autoreg_epoch_14" \
    # --postfix "[force-head]" \
