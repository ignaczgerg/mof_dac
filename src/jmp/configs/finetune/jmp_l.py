"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
from pathlib import Path

from jmp.lightning import GradientClippingConfig

from ...models.gemnet.config import GOCBackboneConfig
from ...tasks.config import AdamWConfig
from ...modules.ema import EMAConfig
from ...tasks.finetune import FinetuneConfigBase
from ...tasks.finetune.base import (
    CheckpointBestConfig,
    RLPConfig,
    WarmupCosRLPConfig
)
from ...tasks.finetune.model_wrapper_base import (
    GemNetModelWrapper, 
)
from ...utils.param_specific_util import make_parameter_specific_optimizer_config

BASE_PATH = Path(__file__).resolve().parent.parent.parent
SMALL_SCALE_FILE_PATH = BASE_PATH / "models/gemnet/scale_files/small.pt"
LARGE_SCALE_FILE_PATH = BASE_PATH / "models/gemnet/scale_files/large.pt"

def jmp_l_ft_config_(
    config: FinetuneConfigBase,
    ckpt_path: Path,
    ema_backbone: bool = True,
    disable_force_output_heads: bool = True,
    args: argparse.Namespace = argparse.Namespace(),
):

    # Assign the model wrapper
    config.model_cls = GemNetModelWrapper
    
    # Assign seed for config and pytorch lightning
    config.trainer.seed = args.seed
    config.set_seed(args.seed)

    # Report MAPE in addition to MAE
    config.metrics.report_mape = True
    config.metrics.report_smape = True

    # Set the model trainer settings for maximum performance
    config.trainer.precision = args.precision
    config.trainer.set_float32_matmul_precision = "medium"
    config.trainer.supports_parameter_hooks = False
    config.trainer.supports_skip_batch_exception = False

    # Set backbone config
    config.backbone = GOCBackboneConfig.large() if args.large else GOCBackboneConfig.small()
    config.embedding.embedding_size = config.backbone.emb_size_atom
    config.backbone.scale_basis = False

    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=args.lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    config.trainer.optimizer.log_grad_norm = True
    config.trainer.optimizer.gradient_clipping = GradientClippingConfig(
        value=1.0,
        algorithm="value",
    )
    # LR Scheduler settings
    warmup_epochs=5 
    warmup_start_lr_factor=1.0e-1
    if args.dataset_name == "qmof":
        warmup_epochs=0 
        warmup_start_lr_factor=1.0
        
    
    config.lr_scheduler = WarmupCosRLPConfig(
        warmup_epochs=warmup_epochs,
        warmup_start_lr_factor=warmup_start_lr_factor,
        should_restart=False,
        max_epochs=32,
        min_lr_factor=0.1,
        rlp=RLPConfig(patience=3, factor=0.8),
    )
    # LLRD Settings
    if args.large:
        max_lr_scales = {
            "embedding": 0.3,
            "blocks_0": 0.55,
            "blocks_1": 0.40,
            "blocks_2": 0.30,
            "blocks_3": 0.40,
            "blocks_4": 0.55,
            "blocks_5": 0.625,
        }
    else:
        max_lr_scales = {
            "embedding": 0.3,
            "blocks_0": 0.35,
            "blocks_1": 0.40,
            "blocks_2": 0.55,
            "blocks_3": 0.625,
        }

    config.parameter_specific_optimizers = make_parameter_specific_optimizer_config(
        config,
        config.backbone.num_blocks,
        max_lr_scales,
    )

    # Checkpoint loading settings
    # We want to use EMA weights from pretraining
    config.meta["ckpt_path"] = ckpt_path
    # if args.large:
    #     config.backbone.scale_file = str(LARGE_SCALE_FILE_PATH)
    # else:
    #     config.backbone.scale_file = str(SMALL_SCALE_FILE_PATH)
    config.meta["ema_backbone"] = ema_backbone

    config.ema = EMAConfig(decay=0.99)
    # Set data config
    config.num_workers = args.num_workers

    # Base early stopping settings
    config.trainer.max_epochs = args.epochs
    config.trainer.max_time = "07:00:00:00"
    # config.early_stopping = EarlyStoppingConfig(
    #     patience=50,
    #     min_delta=1.0e-8,
    #     min_lr=1.0e-8,
    # )
    config.ckpt_best = CheckpointBestConfig()

    # If we are not using force output heads, we need to disable them
    if disable_force_output_heads:
        config.backbone.regress_forces = False
        config.backbone.direct_forces = False
