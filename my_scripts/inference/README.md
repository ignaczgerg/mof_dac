# How to launch predictions

CIF files -> NPZs with properties (CO2 uptake etc).

### Procure an IBEX V100 machine
```bash
srun --mem=20G --ntasks=1 --gres=gpu:v100:1 --cpus-per-task=8 --time=8:00:00 --pty /bin/bash -i
```

### Get the podman image and load it to your local podman state:
```
podman load -i /ibex/project/c2318/software/podman_images/dac_v3_inference.tar
```

### Inside the GPU machine, run the podman container:
```bash
podman run --rm -it \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  localhost/dac:v3_inference bash
```

### Inside the container:

```bash
# Change directory to a container-local copy of Gergo/Yasir's code:
cd /aramco_dac/
mamba activate aramco_dac
export CKPT=/aramco_dac/models/rzq9ckuz/checkpoints/last.ckpt
# Set the input folder with cifs. Modify the export below as you need.
export CIF_DIR=/aramco_dac/datasets/adsorption-mof-ai-generated-batch-1/cifs/
export OUT_DIR=prediction_results/
# Launch the prediction:
bash my_scripts/inference/run_cif_inference.sh
```

It should take 1.5 minutes for 300 CIFs on 1xA100 GPU.

The console output should be like this:

```bash
CKPT: /aramco_dac/models/rzq9ckuz/checkpoints/last.ckpt
CIF_DIR: /aramco_dac/datasets/adsorption-mof-ai-generated-batch-1/cifs/
OUT_DIR: prediction_results/
4677e022ac83
Wed Dec 10 14:13:10 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.15              Driver Version: 570.86.15      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
...
my_scripts/inference/run_cif_inference.sh: line 52: conda: command not found
Unrecognized arguments:  dict_keys(['name', 'regress_energy'])
Freezing 0 parameters (0.00%) out of 30,175,366 total parameters (30,175,366 trainable)
[EMA] Loading 390 EMA parameters into model...
[EMA] EMA weights successfully loaded!
Prefilter CIFs: 100%|███████████████████████████████████████████████████| 307/307 [00:08<00:00, 37.18file/s]
[INFO] CIFs considered: 307 (limit=10000)
Predicting: 100%|███████████████████████████████████████████████████| 307/307 [01:51<00:00,  2.77batch/s]
```

The contents of the results folder should be like this:

```
ls -la prediction_results/
total 48
drwxr-xr-x  2 root root    0 Dec 10 14:15 .
drwxr-xr-x 13 root root    0 Dec 10 14:15 ..
-rw-r--r--  1 root root 3120 Dec 10 14:15 co2_uptake_db_1.npz
-rw-r--r--  1 root root 3107 Dec 10 14:15 co2_uptake_db_2.npz
-rw-r--r--  1 root root 3123 Dec 10 14:15 kh_co2_db_1.npz
-rw-r--r--  1 root root 3114 Dec 10 14:15 kh_co2_db_2.npz
-rw-r--r--  1 root root 3170 Dec 10 14:15 kh_h2o_db_1.npz
-rw-r--r--  1 root root 3155 Dec 10 14:15 kh_h2o_db_2.npz
-rw-r--r--  1 root root 3067 Dec 10 14:15 qst_co2_db_1.npz
-rw-r--r--  1 root root 3069 Dec 10 14:15 qst_co2_db_2.npz
-rw-r--r--  1 root root 3104 Dec 10 14:15 qst_h2o_db_1.npz
-rw-r--r--  1 root root 3105 Dec 10 14:15 qst_h2o_db_2.npz
-rw-r--r--  1 root root 3153 Dec 10 14:15 selectivity_co2_h2o_db_1.npz
-rw-r--r--  1 root root 3148 Dec 10 14:15 selectivity_co2_h2o_db_2.npz
```


## For podman image developer

After updating a running container, commit with:
```bash
podman commit 4677e022ac83 dac:v3_inference
```

Then export to container users with:
```bash
podman save -o dac_v3_inference.tar dac:v3_inference
```

or:

```bash
podman save dac:v3_inference | gzip > dac_v3_inference.tar.gz
# To be loaded via
gunzip -c dac_v3_inference.tar.gz | podman load
```
