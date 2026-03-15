#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name db2
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-user=gergo.ignacz@kaust.edu.sa
#SBATCH --mail-type=FAIL
#SBATCH --time=1:30:00
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

# python preprocesks.py \
#   --csv_paths /ibex/project/c2261/datasets/all-adsorption/DATABASES/MOF-DB-2.0/MOF-DB-2-merged.csv \
#   --cif_path /ibex/project/c2261/datasets/all-adsorption/DATABASES/MOF-DB-2.0/CIF_FILES_V2_1_1 \
#   --entity_key FrameworkName \
#   --backend process \
#   --num_workers 32 \
#   --log_level INFO \
#   --log_every 1000 \
#   --cif_suffixes=_mepoml 


python write_lmdb.py \
  --csv_paths /ibex/project/c2261/datasets/all-adsorption/DATABASES/MOF-DB-2.0/MOF-DB-2-merged.csv \
  --cif_path /ibex/project/c2261/datasets/all-adsorption/DATABASES/MOF-DB-2.0/CIF_FILES_V2_1_1 \
  --entity_key FrameworkName \
  --out_dir /ibex/project/c2261/datasets/all-adsorption/mof_db_2/lmdb \
  --backend process \
  --num_workers 32 \
  --map_size_gb 16 \
  --ratio 94,2,4 \
  --commit_interval 2000 \
  --log_level INFO \
  --log_every 500 \
  --cif_suffixes=_mepoml