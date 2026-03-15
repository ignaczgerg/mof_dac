"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
from pathlib import Path

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...modules.transforms.normalize import PositionNormalizationConfig as PNC

from ...tasks.config import AdamWConfig
from ...tasks.finetune import QMOFConfig
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import (
    EarlyStoppingConfig,
    PrimaryMetricConfig,
    RLPConfig,
    WarmupCosRLPConfig,
)
from ...utils.param_specific_util import (
    make_parameter_specific_optimizer_config,
    make_parameter_specific_optimizer_config_from_optimizer_config,
)

STATS: dict[str, NC] = {
    "y": NC(mean=2.1866251527, std=1.175752521125648),
}

STATS_POS: dict[str, PNC] = {
    "pos": PNC(mean=0.0, std=2.6668860912323),
}


def jmp_l_qmof_config_(config: QMOFConfig, base_path: Path, target: str = "y", args: argparse.Namespace = None):
    # Optimizer settings
    # config.optimizer = AdamWConfig(
    #     lr=5.0e-6,
    #     amsgrad=False,
    #     betas=(0.9, 0.95),
    #     weight_decay=0.1,
    # )

    # Set data config
    config.batch_size = 4

    # Set up dataset
    config.train_dataset = DC.qmof_config(base_path, "train", args=args)
    config.val_dataset = DC.qmof_config(base_path, "val", args=args)
    config.test_dataset = DC.qmof_config(base_path, "test", args=args)

    # Set up normalization
    if (normalization_config := STATS.get(target)) is None:
        raise ValueError(f"Normalization for {target} not found")
    config.normalization = {target: normalization_config}

    # QMOF specific settings
    config.primary_metric = PrimaryMetricConfig(name="y_mae", mode="min")

    # Make sure we only optimize for the single target
    config.graph_scalar_targets = [target]
    config.node_vector_targets = []
    config.graph_classification_targets = []
    # config.graph_scalar_reduction = {target: "sum"}
    config.graph_scalar_reduction_default = "mean"


def equiformer_v2_qmof_config_(config: QMOFConfig, base_path: Path, targets: list[str] = ["y"], args: argparse.Namespace = argparse.Namespace()):
    # Optimizer settings
    # config.optimizer = AdamWConfig(
    #     lr=5.0e-6,
    #     amsgrad=False,
    #     betas=(0.9, 0.95),
    #     weight_decay=0.1,
    # )
    
    target = targets[0] 

    # Set data config
    config.batch_size = args.batch_size

    # Set up dataset
    config.train_dataset = DC.qmof_config(base_path, "train", args=args)
    config.val_dataset = DC.qmof_config(base_path, "val", args=args)
    config.test_dataset = DC.qmof_config(base_path, "test", args=args)

    # Set up normalization
    if (normalization_config := STATS.get(target)) is None:
        raise ValueError(f"Normalization for {target} not found")
    if args.position_norm:
        if target == 'force':
            normalization_config.std = normalization_config.std / STATS_POS['pos'].std
        config.normalization = {target: normalization_config, 'pos': STATS_POS['pos']}
    else:
        config.normalization = {target: normalization_config}

    # QMOF specific settings
    config.primary_metric = PrimaryMetricConfig(name="y_mae", mode="min")

    # Make sure we only optimize for the single target
    config.graph_scalar_targets = [target]
    config.node_vector_targets = []
    config.graph_classification_targets = []
    # config.graph_scalar_reduction = {target: "sum"}
    config.graph_scalar_reduction_default = "mean"


    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=args.lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    # these are taken from official JMP configs 
    # config.lr_scheduler = WarmupCosRLPConfig(
    #     warmup_epochs=5,
    #     warmup_start_lr_factor=1.0e-1,
    #     should_restart=False,
    #     max_epochs=32,
    #     min_lr_factor=0.1,
    #     rlp=RLPConfig(patience=25, factor=0.8),
    # )
    
    # Experiment with different warmup settings
    config.lr_scheduler = WarmupCosRLPConfig(
        warmup_epochs=10,
        warmup_start_lr_factor=1.0e-1,
        should_restart=False,
        max_epochs=args.epochs,
        min_lr_factor=0.1,
        rlp=RLPConfig(patience=3, factor=0.8),
    )
    
    param_optimizer_settings = {
        "no_weight_decay": AdamWConfig(
                lr=args.lr,
                amsgrad=False,
                betas=(0.9, 0.95),
                weight_decay=0.0,
            ), 
    }
    
    # Config backbone
    config.backbone.max_num_elements = 100 # This is the maximum number of elements in the graph
    
    config.parameter_specific_optimizers = make_parameter_specific_optimizer_config_from_optimizer_config(
        config, 
        config.backbone.num_layers, 
        param_optimizer_settings,
    )