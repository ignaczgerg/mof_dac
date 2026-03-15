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
from ...modules.transforms.normalize import PositionNormalizationConfig as PNC
from ...tasks.config import AdamWConfig
from ...tasks.finetune import RMD17Config
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

STATS_POS: dict[str, PNC] = {
    "pos": PNC(mean=0.0, std=2.6668860912323),
}

STATS: dict[str, dict[str, NC]] = {
    "aspirin": {
        "y": NC(mean=-17617.379355234374, std=0.2673998440577667),
        "force": NC(mean=0.0, std=0.2673998440577667),
        # "force": NC(mean=0.0, std=1.2733363),  # why does the std of the force is different from the y.
    },
    "azobenzene": {
        "y": NC(mean=-15553.118351233397, std=0.2866098335926971),
        "force": NC(mean=0.0, std=1.2940075),
    },
    "benzene": {
        "y": NC(mean=-6306.374855859375, std=0.10482645661015047),
        "force": NC(mean=0.0, std=0.90774584),
    },
    "ethanol": {
        "y": NC(mean=-4209.534573266602, std=0.18616576961275716),
        "force": NC(mean=0.0, std=1.1929188),
    },
    "malonaldehyde": {
        "y": NC(mean=-7254.903633896484, std=0.1812291921138577),
        "force": NC(mean=0.0, std=1.302443),
    },
    "naphthalene": {
        "y": NC(mean=-10478.192319667969, std=0.24922674853668708),
        "force": NC(mean=0.0, std=1.3102233),
    },
    "paracetamol": {
        "y": NC(mean=-13998.780924130859, std=0.26963984094801224),
        "force": NC(mean=0.0, std=1.2707518),
    },
    "salicylic": {
        "y": NC(mean=-13472.110348867187, std=0.2437920552529055),
        "force": NC(mean=0.0, std=1.3030343),
    },
    "toluene": {
        "y": NC(mean=-7373.347077485351, std=0.22534282741069667),
        "force": NC(mean=0.0, std=1.246547),
    },
    "uracil": {
        "y": NC(mean=-11266.351949697266, std=0.2227113171300836),
        "force": NC(mean=0.0, std=1.3692871),
    },
}


def jmp_l_rmd17_config_(
    config: RMD17Config, molecules: list[DC.RMD17Molecule], base_path: Path, args: argparse.Namespace = argparse.Namespace()
):
    # TODO: support multiple molecules
    assert len(molecules) == 1, "Only one molecule is supported for rMD17"
    molecule = cast(DC.RMD17Molecule, molecules[0])
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
    config.train_dataset = DC.rmd17_config(molecule, base_path, "train", args=args)
    config.val_dataset = DC.rmd17_config(molecule, base_path, "val", args=args)
    config.test_dataset = DC.rmd17_config(molecule, base_path, "test", args=args)

    # RMD17 specific settings
    config.molecule = molecule
    config.primary_metric = PrimaryMetricConfig(name="force_mae", mode="min")

    # Gradient forces
    config.model_type = "forces"
    config.gradient_forces = True
    config.trainer.inference_mode = False
    config.trainer.precision = "32-true"

    # Set up normalization
    if (normalization_config := STATS.get(molecule)) is None:
        raise ValueError(f"Normalization for {molecule} not found")
    config.normalization = normalization_config

    # We use more conservative early stopping for rMD17
    #   (we essentially copy Allegro here).
    config.trainer.max_epochs = args.epochs
    config.trainer.max_time = "07:00:00:00"
    # config.early_stopping = EarlyStoppingConfig(
    #     patience=1000,
    #     min_delta=1.0e-8,
    #     min_lr=1.0e-10,
    # )

    # We also use a conservative set of hyperparameters
    #   for ReduceLROnPlateau (again, we copy Allegro here).
    # The main difference is that we use a larger patience (25 vs 3).
    config.lr_scheduler = WarmupCosRLPConfig(
        warmup_epochs=0,#5,
        warmup_start_lr_factor=1.0,#1.0e-1,
        should_restart=False,
        max_epochs=32,
        min_lr_factor=0.1,
        rlp=RLPConfig(patience=25, factor=0.8),
    )


def equiformer_v2_rmd17_config_(
    config: RMD17Config, molecules: list[DC.RMD17Molecule], base_path: Path, args: argparse.Namespace = argparse.Namespace()
):
    # TODO: support multiple molecules
    assert len(molecules) == 1, "Only one molecule is supported for rMD17"
    molecule = cast(DC.RMD17Molecule, molecules[0])
    
    
    '''
    Config for EquiformerV1 on MD17:
        alpha_drop: 0.0
        basis_type: exp
        drop_path_rate: 0.0
        fc_neurons:
        - 64
        - 64
        irreps_equivariant_inputs: 1x0e+1x1e+1x2e
        irreps_feature: 512x0e+256x1e+128x2e
        irreps_head: 32x0e+16x1e+8x2e
        irreps_in: 64x0e
        irreps_mlp_mid: 384x0e+192x1e+96x2e
        irreps_node_attr: 1x0e
        irreps_node_embedding: 128x0e+64x1e+32x2e
        irreps_pre_attn: 128x0e+64x1e+32x2e
        irreps_sh: 1x0e+1x1e+1x2e
        max_radius: 5.0
        name: equiformer_v1_denoising_pos_md17_v2
        nonlinear_message: true
        norm_layer: layer
        num_heads: 4
        num_layers: 6
        number_of_basis: 32
        out_drop: 0.0
        proj_drop: 0.0
        rescale_degree: false
    '''

    # Set data config
    config.batch_size = args.batch_size           # 8 for EqV1, 4 for JMP
    
    config.backbone.max_radius = 7.0
    config.backbone.max_neighbors = 100



    # Set up dataset
    config.train_dataset = DC.rmd17_config(molecule, base_path, "train", args=args)
    config.val_dataset = DC.rmd17_config(molecule, base_path, "val", args=args)
    config.test_dataset = DC.rmd17_config(molecule, base_path, "test", args=args)

    # RMD17 specific settings
    config.molecule = molecule
    config.primary_metric = PrimaryMetricConfig(name="force_mae", mode="min")



    # Changing the ratio of coefficients
    config.graph_scalar_loss_coefficients = {"y": 1.0}
    config.node_vector_loss_coefficients = {"force": 80.0}
    
    # Gradient forces
    config.model_type = "energy_forces" # # energy_forces (two head one for energy and one for forces, when the gradient is false. it will supervise both) before it was forces because it only supervised forces.
    config.gradient_forces = False # use the force head for faster convergence
    config.trainer.inference_mode = False
    config.trainer.precision = "32-true"

    # Set up normalization
    if (normalization_config := STATS.get(molecule)) is None:
        raise ValueError(f"Normalization for {molecule} not found")
    if args.position_norm:
        for target, stats in normalization_config.items():
            if target == 'force':
                stats.std = stats.std / STATS_POS['pos'].std
        normalization_config['pos'] = STATS_POS['pos']
    config.normalization = normalization_config

    # We use more conservative early stopping for rMD17
    #   (we essentially copy Allegro here).
    config.trainer.max_epochs = args.epochs
    config.trainer.max_time = "07:00:00:00"
    # config.early_stopping = EarlyStoppingConfig(
    #     patience=1000,
    #     min_delta=1.0e-8,
    #     min_lr=1.0e-10,
    # )

    # We also use a conservative set of hyperparameters
    #   for ReduceLROnPlateau (again, we copy Allegro here).
    # The main difference is that we use a larger patience (25 vs 3).
    # config.lr_scheduler = WarmupCosRLPConfig(
    #     warmup_epochs=0,#5,
    #     warmup_start_lr_factor=1.0,#1.0e-1,
    #     should_restart=False,
    #     max_epochs=32,
    #     min_lr_factor=0.1,
    #     rlp=RLPConfig(patience=25, factor=0.8),
    # )

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
    # config.lr_scheduler = WarmupCosRLPConfig(
    #     warmup_epochs=30,
    #     warmup_start_lr_factor=1.0e-1,
    #     should_restart=False,
    #     max_epochs=750, #300
    #     min_lr_factor=0.1,
    #     rlp=RLPConfig(patience=25, factor=0.8),
    # )
    
    # # Experiment with different warmup settings
    # config.lr_scheduler = WarmupCosRLPConfig(
    #     warmup_epochs=30,
    #     warmup_start_lr_factor=1.0e-1,
    #     should_restart=False,
    #     max_epochs=300,
    #     min_lr_factor=0.1,
    #     rlp=RLPConfig(patience=25, factor=0.8),
    # )

    # Experiment with the same hyperparameters as Eqv1
    config.lr_scheduler = WarmupCosRLPConfig(
        warmup_epochs=10,
        warmup_start_lr_factor=0.002,   # still starts at 0.002 * base LR
        should_restart=False,
        max_epochs=1500,
        min_lr_factor=0.002,            # still ends at 0.002 * base LR
        rlp=RLPConfig(
            patience=10,
            factor=0.1
        ),
    )
    
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