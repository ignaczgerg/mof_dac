#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name=cif-infer
#SBATCH -o output/infer.%A.out
#SBATCH -e output/infer.%A.err
#SBATCH --mail-user=${USER}@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=29G
#SBATCH --cpus-per-task=2



CKPT="/ibex/project/c2261/dac_iclr/finetune/lightning_logs/yozti0u8/aramco_dac_finetune/yozti0u8/checkpoints/last.ckpt"
# CIF_DIR="/home/ignaczg/projects/aramco_dac/temp"
# CIF_DIR="/home/ignaczg/projects/aramco_dac/temp/Firas/batch_5/"
# OUT_DIR="/ibex/project/c2261/dac_iclr/predictions/firas/batch_5/"
CIF_DIR="/ibex/project/c2261/datasets/adsorption-mof-ai-generated-batch-3/cif"
OUT_DIR="/ibex/project/c2261/dac_iclr/predictions/adsorption-mof-ai-generated-batch-3/"
LIST_FILE=""
LIST_COL=""

BATCH_SIZE=1
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-2}
DEVICE="cuda"      
PATTERN=".*\.cif$"
SID_FROM="stem"
FAIL_ON_ERROR=0
MAX_ATOMS=3000
LIMIT_FILES=10000

hostname
nvidia-smi
conda activate aramco_dac


mkdir -p "$(dirname "$OUT_DIR")" output

EXTRA_FLAGS=()
if [[ -n "${LIST_FILE}" ]]; then
  EXTRA_FLAGS+=(--list_file "${LIST_FILE}")
  if [[ -n "${LIST_COL}" ]]; then
    EXTRA_FLAGS+=(--list_col "${LIST_COL}")
  fi
fi
if [[ "${FAIL_ON_ERROR}" == "1" ]]; then
  EXTRA_FLAGS+=(--fail_on_error)
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# Best practice is to run the script from the root of the repository.
python my_scripts/inference/cif_inference.py \
  --ckpt "${CKPT}" \
  --cif_dir "${CIF_DIR}" \
  --pattern "${PATTERN}" \
  --sid_from "${SID_FROM}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --out "${OUT_DIR}" \
  --max_atoms 3000 \
  --prefilter_workers 8 \
  --limit_files 10000 \
  "${EXTRA_FLAGS[@]}" \