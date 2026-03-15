# Aramco DAC Project

This repository contains the training setup for pretraining and fine-tuning models used in the Aramco DAC project for adsorption and MOF property prediction.

---

## Pretraining

Pretraining is performed on large-scale datasets (e.g., ANI-1x, Transition-1x, OC20, OC22).

To launch pretraining, run:

```bash
cd my_scripts/run_exp/
sbatch run_pretrain.sh
```

---

## Fine-Tuning

Fine-tuning is performed on MOF and adsorption datasets.

To launch fine-tuning, run:

```bash
cd my_scripts/run_exp/
sbatch run_finetune_mof_db.sh
```

You can modify the target variable in the script to select from:

```
['qst_co2', 'qst_h2o', 'qst_n2',
 'kh_co2', 'kh_h2o', 'kh_n2',
 'selectivity_co2_h2o', 'selectivity_co2_n2',
 'co2_uptake', 'n2_uptake']
```

---

## Installation

Set up the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml -n aramco_dac
conda activate aramco_dac
pip install -e .
```

---

## Notes
- Ensure dataset paths and config files are correctly set before running `sbatch`.
- Pretraining and fine-tuning scripts are located in `my_scripts/run_exp/`.
---
