#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name cof2
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=40:30:00
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


# this is just to test if the preprocessing works or not.
# python preprocess.py \
#   --csv_paths /ibex/project/c2261/datasets/all-adsorption/DATABASES/COF-DB-2.0/COF-DB-2.0-Adsorption-v2.1.2.csv \
#   --cif_path /ibex/project/c2261/datasets/all-adsorption/DATABASES/COF-DB-2.0/CIF_FILES \
#   --entity_key FrameworkName \
#   --backend process \
#   --num_workers 1 \
#   --log_level INFO \
#   --log_every 1000 \
#   --cif_suffixes=_mepoml 


python write_lmdb.py \
  --csv_paths /ibex/project/c2261/datasets/all-adsorption/DATABASES/COF-DB-2.0/COF-DB-2.0-Adsorption-v2.1.2.csv \
  --cif_path /ibex/project/c2261/datasets/all-adsorption/DATABASES/COF-DB-2.0/CIF_FILES \
  --entity_key FrameworkName \
  --out_dir /ibex/project/c2261/datasets/all-adsorption/cof_db_2/lmdb \
  --backend process \
  --num_workers 32 \
  --map_size_gb 16 \
  --ratio 877,41,82 \
  --commit_interval 2000 \
  --log_level INFO \
  --log_every 500 \