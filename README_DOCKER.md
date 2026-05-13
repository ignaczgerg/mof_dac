# Aramco DAC Docker Build Guide

This guide explains how to build the **Aramco DAC** Docker image.

---

## 1. Build a Docker image
```bash
# Docker
sudo docker build -t aramco_dac_env .
# Podman
sudo podman build -t aramco_dac_env .
```

This builds a fully self-contained image including:
  - Ubuntu 22.04 + CUDA 12.1 + cuDNN 8  
  - Miniconda with the environment defined in `environment.yml`  
  - The full project source code under `/opt/app`  

First-time build may take 15–30 minutes depending on your machine.
---

## 2. Save the image as a portable `.tar` file

```bash
# Docker
sudo docker save -o aramco_dac_env.tar aramco_dac_env
sudo chown $(whoami):$(id -gn) aramco_dac_env.tar

# Podman
sudo podman save -o aramco_dac_env_podman.tar aramco_dac_env
sudo chown $(whoami):$(id -gn) aramco_dac_env_podman.tar

```
---
## 3. Transfer file

Once the image is built and saved, compress and transfer:

```bash
# Docker
zip -r aramco_dac_env_package.zip aramco_dac_env.tar README.md
# Podman
zip -r aramco_dac_env_package_podman.zip aramco_dac_env_podman.tar README.md

# Transfer compressed
scp aramco_dac_env_package_podman.zip Ibex:/ibex/ai/project/c2261/podman_images/
# Transfer uncompressed
scp aramco_dac_env_podman.tar Ibex:/ibex/ai/project/c2261/podman_images/
```

## 3.1 Additional Setup for Podman Users on local machines (GPU Support)

If you are using **Podman**, you must enable NVIDIA GPU
support on your system before running the container:

```bash
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=crun
sudo systemctl restart podman
```

---
## 4. In the cluster, run the container
Mount the resources (datasets + checkpoints) and output directories from the host into the container:

```bash
# Docker
sudo docker load -i aramco_dac_env.tar
sudo docker run --gpus all -it \
  -v /path/on/host/resources:/opt/resources \
  -v /path/on/host/output:/opt/output \
  -v /path/on/host/run_finetune_mof_db_test.sh:/opt/app/my_scripts/run_exp/run_finetune_mof_db_test.sh \
  aramco_dac_env 

# Podman
mkdir -p /run/user/${UID}/bus /ibex/user/$USER/podman_images/
podman load --root=/ibex/user/$USER/podman_images -i aramco_dac_env_podman.tar

podman run --rm -it \
  --root=/ibex/user/$USER/podman_images \
  --device nvidia.com/gpu=all \
  -v /path/on/host/resources:/opt/resources \
  -v /path/on/host/output:/opt/output \
  -v /path/on/host/run_finetune_mof_db_test.sh:/opt/app/my_scripts/run_exp/run_finetune_mof_db_test.sh \
  aramco_dac_env

# Example for valid paths in Ibex:
podman run --rm -it \
  --root=/ibex/user/$USER/podman_images \
  --device nvidia.com/gpu=all \
  -v /ibex/ai/project/c2261:/opt/resources \
  -v /ibex/ai/project/c2261/podman_images/image_18_Nov_2025/output:/opt/output \
  -v /ibex/ai/project/c2261/podman_images/image_18_Nov_2025/run_finetune_mof_db_test.sh:/opt/app/my_scripts/run_exp/run_finetune_mof_db_test.sh \
  aramco_dac_env
```

Then inside docker, run:
```bash
cd /opt/app/my_scripts/run_exp
bash run_finetune_mof_db_test.sh
```

## Directory mapping:
- (image containing source code)    -> /opt/app       (project source code and scripts)
- /path/on/host/resources           -> /opt/resources (datasets and pretrained checkpoints)
- /path/on/host/output              -> /opt/output    (logs and experiment results)