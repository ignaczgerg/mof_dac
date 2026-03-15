#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name ads_db2_lmdb
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=1:30:00
### SBATCH --gres=gpu:v100:1
#SBATCH --mem=300G
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
  python db2_1_write_lmdb.py \
    --backend process --num_workers 32 \
  --dataset_path /ibex/project/c2261/datasets/adsorption_db2_merged \
  --entity_key HMOF_ID \
  --out_dir /ibex/project/c2261/datasets/adsorption_db2_merged/lmdb \
  --map_size_gb 24 --commit_interval 10000 --log_level INFO --ratio 94,2,4
