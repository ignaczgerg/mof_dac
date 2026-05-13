# =========================================================
#  Aramco DAC - Self-contained offline Docker environment
# =========================================================

# --- Base CUDA image with cuDNN ---
FROM docker.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# --- System dependencies ---
RUN apt-get update && apt-get install -y \
    wget git build-essential libgl1-mesa-glx libglib2.0-0 nano && \
    rm -rf /var/lib/apt/lists/*

# --- Install Miniconda ---
WORKDIR /opt
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH

# --- Working directory for your project ---
WORKDIR /opt/app

# --- Copy your full project (including environment.yml) ---
COPY . /opt/app

# --- Remove all defaults and define only the required channels defined in environment.yml ---
RUN echo "channels:"                >  /opt/conda/.condarc && \
    echo "  - pyg"                  >> /opt/conda/.condarc && \
    echo "  - nvidia"               >> /opt/conda/.condarc && \
    echo "  - pytorch"              >> /opt/conda/.condarc && \
    echo "  - conda-forge"          >> /opt/conda/.condarc && \
    cat /opt/conda/.condarc && \
    conda config --show channels

# --- Create Conda environment (installs both conda + pip deps) ---
RUN conda env create -f /opt/app/environment.yml && \
    conda clean -afy

# --- Activate environment for following RUN commands ---
SHELL ["conda", "run", "-n", "aramco_dac", "/bin/bash", "-c"]

# --- Install your project (editable mode, offline-safe) ---
RUN pip install -e /opt/app

# --- Create mount points for data and output ---
RUN mkdir -p /opt/resources /opt/output

# --- Default entrypoint (always use the Conda environment) ---
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "aramco_dac", "bash"]