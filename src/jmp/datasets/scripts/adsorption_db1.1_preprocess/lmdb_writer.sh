#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name ads_db2_lmdb
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=120:30:00
### SBATCH --gres=gpu:v100:1
#SBATCH --mem=600G
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --ntasks=1

conda activate aramco_dac
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun --cpu-bind=cores \
  python db1_1_write_lmdb.py \
    --backend process --num_workers 32 \
    --dataset_path /ibex/project/c2261/datasets/adsorption_db1_merged \
    --out_dir /ibex/project/c2261/datasets/adsorption_db1_merged/lmdb \
    --entity_key HMOF_ID \
    --map_size_gb 2 --commit_interval 1000 --ratio 85,5,15

  
# python db1_1_write_lmdb.py \
#   --dataset_path /ibex/project/c2261/datasets/adsorption_db1_merged \
#   --entity_key HMOF_ID \
#   --out_dir /ibex/project/c2261/datasets/adsorption_db1_merged/lmdb \
#   --num_workers 32 --map_size_gb 2 --commit_interval 1000 --log_level INFO --ratio 85,5,15
