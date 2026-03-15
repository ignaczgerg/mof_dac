"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
from pathlib import Path
from typing import cast

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...tasks.config import AdamWConfig
from ...tasks.finetune import MD22Config
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

MD22_MOLECULES: list[tuple[DC.MD22Molecule, int, bool, bool]] = [
    ("Ac-Ala3-NHMe", 2, False, True),
    ("DHA", 1, True, False),
    ("stachyose", 1, True, True),
    ("stachyose", 1, True, True),
    ("AT-AT", 1, True, True),
    ("AT-AT-CG-CG", 1, True, True),
    ("buckyball-catcher", 1, True, True),
    ("double-walled_nanotube", 1, False, True),
]

STATS: dict[str, dict[str, NC]] = {
    "Ac-Ala3-NHMe": {
        "y": NC(mean=-26913.953, std=0.35547638),
        "force": NC(mean=0.0, std=1.1291506),
    },
    "DHA": {
        "y": NC(mean=-27383.035, std=0.41342595),
        "force": NC(mean=0.0, std=1.1258113),
    },
    "stachyose": {
        "y": NC(mean=-68463.59, std=0.5940788),
        "force": NC(mean=0.0, std=1.1104717),
    },
    "AT-AT": {
        "y": NC(mean=-50080.08, std=0.47309175),
        "force": NC(mean=0.0, std=1.2109985),
    },
    "AT-AT-CG-CG": {
        "y": NC(mean=-101034.23, std=0.680055),
        "force": NC(mean=0.0, std=1.2021886),
    },
    "buckyball-catcher": {
        "y": NC(mean=-124776.7, std=0.64662045),
        "force": NC(mean=0.0, std=1.0899031),
    },
    "double-walled_nanotube": {
        "y": NC(mean=-338224.16, std=3.3810701),
        "force": NC(mean=0.0, std=1.0137014),
    },
}


def jmp_l_md22_config_(
    config: MD22Config,
    molecules: list[DC.MD22Molecule],
    base_path: Path,
    args: argparse.Namespace = argparse.Namespace()
):
    # TODO: support multiple molecules
    assert len(molecules) == 1, "Only one molecule is supported for MD22"
    molecule = cast(DC.MD22Molecule, molecules[0])

    # Optimizer settings
    # config.optimizer = AdamWConfig(
    #     lr=5.0e-6,
    #     amsgrad=False,
    #     betas=(0.9, 0.95),
    #     weight_decay=0.1,
    # )

    batch_size = None
    is_grad = None
    amp = None
    for target_name, batch_size, is_grad, amp in MD22_MOLECULES:
        if molecule == target_name:
            batch_size = batch_size
            is_grad = is_grad
            amp = amp
            break
    
    # Set data config
    if args.batch_size:
        batch_size = args.batch_size
    config.batch_size = batch_size

    # Set up dataset
    config.train_dataset = DC.md22_config(molecule, base_path, "train", args=args)
    config.val_dataset = DC.md22_config(molecule, base_path, "val", args=args)
    config.test_dataset = DC.md22_config(molecule, base_path, "test", args=args)

    # MD22 specific settings
    config.molecule = molecule
    config.primary_metric = PrimaryMetricConfig(name="force_mae", mode="min")

    # Gradient forces
    config.model_type = "forces"
    if is_grad:
        config.gradient_forces = True
        config.trainer.inference_mode = False
    else:
        config.backbone.regress_forces = True
        config.backbone.direct_forces = True

    if amp:
        config.trainer.precision = "16-mixed"
    else:
        config.trainer.precision = "32-true"

    # Set up normalization
    if (normalization_config := STATS.get(molecule)) is None:
        raise ValueError(f"Normalization for {molecule} not found")
    config.normalization = normalization_config


def equiformer_v2_md22_config_(
    config: MD22Config,
    molecules: list[DC.MD22Molecule],
    base_path: Path,
    args: argparse.Namespace = argparse.Namespace()
):
    # TODO: support multiple molecules
    assert len(molecules) == 1, "Only one molecule is supported for MD22"
    molecule = cast(DC.MD22Molecule, molecules[0])
    
    config.cutoff = 12.0
    config.max_neighbors = 30
    config.backbone.max_radius = config.cutoff
    config.backbone.max_neighbors = config.max_neighbors

    batch_size = None
    is_grad = None
    amp = None
    for target_name, batch_size, is_grad, amp in MD22_MOLECULES:
        if molecule == target_name:
            batch_size = batch_size
            is_grad = is_grad
            amp = amp
            break
    
    # Set data config
    if args.batch_size:
        batch_size = args.batch_size
    config.batch_size = batch_size

    # Set up dataset
    config.train_dataset = DC.md22_config(molecule, base_path, "train", args=args)
    config.val_dataset = DC.md22_config(molecule, base_path, "val", args=args)
    config.test_dataset = DC.md22_config(molecule, base_path, "test", args=args)

    # MD22 specific settings
    config.molecule = molecule
    config.primary_metric = PrimaryMetricConfig(name="force_mae", mode="min")

    # Gradient forces
    config.model_type = "forces"
    if is_grad:
        config.gradient_forces = True
        config.trainer.inference_mode = False
    else:
        config.backbone.regress_forces = True
        config.backbone.direct_forces = True

    if amp:
        config.trainer.precision = "16-mixed"
    else:
        config.trainer.precision = "32-true"

    # Set up normalization
    if (normalization_config := STATS.get(molecule)) is None:
        raise ValueError(f"Normalization for {molecule} not found")
    config.normalization = normalization_config


    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=args.lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    # these are taken from official JMP configs 
    config.lr_scheduler = WarmupCosRLPConfig(
        warmup_epochs=5,
        warmup_start_lr_factor=1.0e-1,
        should_restart=False,
        max_epochs=32,
        min_lr_factor=0.1,
        rlp=RLPConfig(patience=3, factor=0.8),
    )
    
    # Experiment with different warmup settings
    # config.lr_scheduler = WarmupCosRLPConfig(
    #     warmup_epochs=30,
    #     warmup_start_lr_factor=1.0e-1,
    #     should_restart=False,
    #     max_epochs=args.epochs,
    #     min_lr_factor=0.1,
    #     rlp=RLPConfig(patience=25, factor=0.8),
    # )
    
    param_optimizer_settings = {
        "no_weight_decay": AdamWConfig(
                lr=args.lr,
                amsgrad=False,
                betas=(0.9, 0.95),
                weight_decay=0.0,
            ), 
    }
    
    config.parameter_specific_optimizers = make_parameter_specific_optimizer_config_from_optimizer_config(
        config, 
        config.backbone.num_layers,  
        param_optimizer_settings,
    )