"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
from pathlib import Path
from typing import cast
import math 

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...modules.transforms.normalize import PositionNormalizationConfig as PNC
from ...tasks.config import AdamWConfig
from ...tasks.finetune import QM9Config
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import PrimaryMetricConfig, RLPConfig, WarmupCosRLPConfig
from ...tasks.finetune.qm9 import QM9Target, SpatialExtentConfig, DefaultOutputHeadConfig


STATS_POS: dict[str, PNC] = {
    "pos": PNC(mean=0.0, std=2.6668860912323),
}

STATS: dict[str, NC] = {
    "mu": NC(mean=2.674587, std=1.5054824),
    "alpha": NC(mean=75.31013, std=8.164021),
    "eps_HMO": NC(mean=-6.5347567, std=0.59702325),
    "eps_LUMO": NC(mean=0.323833, std=1.273586),
    "delta_eps": NC(mean=6.8585854, std=1.283122),
    "R_2_Abs": NC(mean=1189.6819, std=280.0421),
    "ZPVE": NC(mean=-0.00052343315, std=0.04904531),
    "U_0": NC(mean=0.0028667436, std=1.0965848),
    "U": NC(mean=0.0028711546, std=1.0941933),
    "H": NC(mean=0.0029801112, std=1.0942822),
    "G": NC(mean=0.000976671, std=1.101572),
    "c_v": NC(mean=-0.005799451, std=2.2179737),
    "U_0_ATOM": NC(mean=-76.15232, std=10.309152),
    "U_ATOM": NC(mean=-76.6171, std=10.400515),
    "H_ATOM": NC(mean=-77.05511, std=10.474532),
    "G_ATOM": NC(mean=-70.87026, std=9.484609),
    "A": NC(mean=11.58375, std=2046.5049),
    "B": NC(mean=1.40327, std=1.1445134),
    "C": NC(mean=1.1256535, std=0.85679144),
}


def jmp_l_qm9_config_(config: QM9Config, targets: list[QM9Target], base_path: Path, args: argparse.Namespace = argparse.Namespace()):
    config.batch_size = 48
    config.backbone.max_radius = 5.0
    config.backbone.max_neighbors = 30


    # Set up dataset
    config.train_dataset = DC.qm9_config(base_path, "train", args=args)
    config.val_dataset = DC.qm9_config(base_path, "val", args=args)
    config.test_dataset = DC.qm9_config(base_path, "test", args=args)
    
    config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_mae", mode="min")

    config.graph_scalar_targets = cast(list[str], targets)
    for target in targets:
        # Set up normalization
        if (normalization_config := STATS.get(target)) is None:
            raise ValueError(f"Normalization for {target} not found")
        config.normalization = {target: normalization_config}

        # Handle R_2_Abs separately
        if target == "R_2_Abs":
            config.output_head = SpatialExtentConfig()
            # Also, we don't use any normalization for this target
            config.normalization = {}
        else:
            config.output_head = DefaultOutputHeadConfig()



def escaip_small_qm9_config_(config: QM9Config, targets: list[QM9Target], base_path: Path, args: argparse.Namespace = argparse.Namespace()):

    config.batch_size = 32
    
    # Set the backbone config
    config.backbone.batch_size = config.batch_size
    config.backbone.max_radius = 8.0
    config.backbone.max_neighbors = 10
    config.backbone.use_pbc = False
    config.backbone.max_num_elements = 5 # only C, N, O, H, F
    '''
    here: num_nodes <= max_num_nodes_per_structure * batch_size = 29 * batch_size
    pad_size = max_num_nodes_per_batch * batch_size - num_nodes
    -> max_num_nodes_per_batch >= num_nodes / batch_size for pad_size >= 0
    '''
    # TODO: Find the rigth configs for qm9
    config.backbone.max_num_nodes_per_batch = 29
    
    # Set up dataset
    config.train_dataset = DC.qm9_config(base_path, "train", args=args)
    config.val_dataset = DC.qm9_config(base_path, "val", args=args)
    config.test_dataset = DC.qm9_config(base_path, "test", args=args)
    
    config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_mae", mode="min")

    config.graph_scalar_targets = cast(list[str], targets)
    for target in targets:
        # Set up normalization
        if (normalization_config := STATS.get(target)) is None:
            raise ValueError(f"Normalization for {target} not found")
        config.normalization = {target: normalization_config}

        # Handle R_2_Abs separately
        if target == "R_2_Abs":
            config.output_head = SpatialExtentConfig()
            # Also, we don't use any normalization for this target
            config.normalization = {}
        else:
            config.output_head = DefaultOutputHeadConfig()



def equiformer_v2_qm9_config_(config: QM9Config, targets: list[QM9Target], base_path: Path, args: argparse.Namespace = argparse.Namespace()):
    config.backbone.use_pbc = False
    config.batch_size = args.batch_size
    # config.batch_size = 48
    config.backbone.max_radius = 12.0
    # config.backbone.max_radius = 5.0        # TO MATCH EquiformerV1 paper
    config.backbone.max_neighbors = 500
    # config.backbone.max_neighbors = 1000     # TO MATCH EquiformerV1 paper

    # # QM9
    # _MAX_ATOM_TYPE = 5
    # # Statistics of QM9 with cutoff radius = 5
    # _AVG_NUM_NODES = 18.03065905448718
    # _AVG_DEGREE = 15.57930850982666

    # # config.backbone.max_num_elements = _MAX_ATOM_TYPE  # only C, N, O, H, F
    # config.backbone.avg_num_nodes = _AVG_NUM_NODES # TO MATCH EquiformerV1 paper
    # config.backbone.avg_degree = _AVG_DEGREE # TO MATCH EquiformerV1 paper

    # Match EquiformerV2 paper settings
    if config.backbone.num_layers == 5:
        config.trainer.precision = "32-true" #, run out of memory with 8 layers 

    # Set up dataset
    config.train_dataset = DC.qm9_config(base_path, "train", args=args)
    config.val_dataset = DC.qm9_config(base_path, "val", args=args)
    config.test_dataset = DC.qm9_config(base_path, "test", args=args)
    
    config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_mae", mode="min")

    config.graph_scalar_targets = cast(list[str], targets)
    for target in targets:
        # Set up normalization
        if (normalization_config := STATS.get(target)) is None:
            raise ValueError(f"Normalization for {target} not found")
        if args.position_norm:
            if target == 'force':
                normalization_config.std = normalization_config.std / STATS_POS['pos'].std
            config.normalization = {target: normalization_config, 'pos': STATS_POS['pos']}
        else:
            config.normalization = {target: normalization_config}
        # config.normalization = {target: normalization_config}

        # Handle R_2_Abs separately
        if target == "R_2_Abs":
            config.output_head = SpatialExtentConfig()
            # Also, we don't use any normalization for this target
            config.normalization = {}
        else:
            config.output_head = DefaultOutputHeadConfig()
    
    # config.lr_scheduler = WarmupCosRLPConfig(
    #     warmup_epochs=5,  # 5
    #     warmup_start_lr_factor=0.002,  # 1e-6 / 5e-4 = 0.002
    #     should_restart=False,  # no restart in the argparse snippet
    #     max_epochs=args.epochs,  # total epochs from your script args
    #     min_lr_factor=0.002,  # 1e-6 / 5e-4 = 0.002
    #     rlp=RLPConfig(
    #         patience=10,  # 10
    #         factor=0.1          # 0.1
    #     ),
    # )