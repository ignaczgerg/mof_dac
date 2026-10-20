# Benchmark Methods Used in This Work

This document lists the external methods/frameworks used for benchmarking.

## 1) PMTransformer (MOFTransformer)
- **Repository:** https://github.com/hspark1212/MOFTransformer
- **Description:** A pretrained multimodal transformer framework for porous materials (including MOFs), combining local atom/graph information and global energy-grid style representations, then fine-tuned for downstream property prediction.

## 2) MOFormer
- **Repository:** https://github.com/zcao0420/MOFormer
- **Description:** A self-supervised transformer for MOF property prediction that uses a structure-agnostic MOF text representation (MOFid), with pretraining and downstream fine-tuning for target properties.
- **MOF representation used in our experiments:** MOFid.
- **MOFid repository:** https://github.com/snurr-group/mofid

## 3) eSEN (via FAIR-Chem)
- **Repository:** https://github.com/facebookresearch/fairchem
- **Description:** The FAIR-Chem codebase used to access and evaluate modern chemistry/materials ML models (including eSEN-related experiments) for structure-based property modeling and benchmarking.

## 4) H2O AutoML
- **Repository:** https://github.com/h2oai/h2o-3
- **Description:** Automated machine learning baseline that trains and ranks multiple model families with standardized workflows for strong tabular benchmarking.
- **Feature engineering (used with H2O AutoML):** SOAP descriptors extracted from CIF files using DScribe.
- **DScribe repository:** https://github.com/SINGROUP/dscribe

## 5) SOAP Descriptors (Feature Engineering)
- **Implementation used:** DScribe SOAP
- **Repository:** https://github.com/SINGROUP/dscribe
- **Description:** Smooth Overlap of Atomic Positions (SOAP) descriptors were computed from CIF crystal structures to generate fixed-length, structure-aware feature vectors for machine learning.
- **Role in this work:** Used as the input feature representation for tabular baselines, including the H2O AutoML benchmark.
