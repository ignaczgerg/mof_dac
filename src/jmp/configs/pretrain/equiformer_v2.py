"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import argparse
import torch
from jmp.lightning import GradientClippingConfig

from ...models.equiformer_v2.config import (
    EquiformerV2Config, 
    AutoregressiveLlamaEquiformerV2Config
)

from ...utils.param_specific_util import (
    pretrain_make_parameter_specific_optimizer_config_,
)

from ...modules.dataset.concat_dataset import MTDatasetConfig
from ...modules.ema import EMAConfig
from ...tasks.config import AdamWConfig
from ...tasks.pretrain import PretrainConfig
from ...tasks.pretrain.module import (
    LinearWarmupCosineAnnealingSchedulerConfig,
    WarmupCosineSchedulerConfig,
    WarmupCosineLambdaLRSchedulerConfig
)
from ...models.gemnet.config import GOCBackboneConfig
from ...tasks.finetune.base import (
    PrimaryMetricConfig,
    CheckpointBestConfig,
)
from ...tasks.pretrain.model_wrapper_base import(
    EquiformerV2ModelWrapper, 
    AutoregressiveEquiformerV2ModelWrapper
)


def equiformer_v2_pt_config_(config: PretrainConfig, args: argparse.Namespace):
    
    # Assing the model wrapper 
    config.model_cls = EquiformerV2ModelWrapper
    
    # Assign seed for config and pytorch lightning
    config.trainer.seed = args.seed
    config.set_seed(args.seed)

    # Assign validation check interval
    config.trainer.val_check_interval = 0.2 # Run validation 5 times per epoch (every 20% of training)
    
    # Set the model trainer settings for maximum performance
    config.trainer.precision = "16-mixed"
    config.trainer.set_float32_matmul_precision = "medium"
    config.trainer.supports_parameter_hooks = False
    config.trainer.supports_skip_batch_exception = False

    # Batch size
    config.batch_size = args.batch_size
    # config.eval_batch_size = 20

    # Epochs
    config.trainer.max_epochs = args.epochs


    # Set backbone config
    if args.medium:
        config.backbone = EquiformerV2Config.medium()
    elif args.large:
        config.backbone = EquiformerV2Config.large()
    else:
        config.backbone = EquiformerV2Config.base()
        

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
        algorithm="norm",
    )
    # LR Scheduler settings
    config.lr_scheduler = LinearWarmupCosineAnnealingSchedulerConfig(
        warmup_steps=2000,
        warmup_start_lr_factor=0.2,
        min_lr_factor=0.1,
        max_epochs=args.epochs, # Note that we set max_epochs to None in the pretrain.py as we use max_steps instead.
    )

    # override default settings to handle large atoms
    config.backbone.max_num_elements = 100

    # EMA settings
    config.ema = EMAConfig(decay=0.99)

    # Set data config
    config.num_workers = args.num_workers

    config.primary_metric = PrimaryMetricConfig(name="loss", mode="min")
    config.ckpt_best = CheckpointBestConfig()

    # Set up the JMP MT dataset config and tasks
    config.mt_dataset = MTDatasetConfig(
        sample_type="temperature",
        sample_temperature=1.0,
    )
    
    
    
def autoreg_equiformer_v2_pt_config_(config: PretrainConfig, args: argparse.Namespace):
    
    # Assing the model wrapper 
    config.model_cls = AutoregressiveEquiformerV2ModelWrapper
    
    # Here we disable force heads for autoregressive pretraining 
    config.regress_forces = False
    
    '''
    optim:
    batch_size:                   16         # 6
    eval_batch_size:              16        # 6
    grad_accumulation_steps:      1 # 64         # gradient accumulation: effective batch size = `grad_accumulation_steps` * `batch_size` * (num of GPUs)
    load_balancing: atoms
    num_workers: 8
    lr_initial:                   0.0004    # [0.0002, 0.0004], eSCN uses 0.0008 for batch size 96

    optimizer: AdamW
    optimizer_params:
        weight_decay: 0.001
    scheduler: LambdaLR
    scheduler_params:
        lambda_type: cosine
        warmup_factor: 0.1     # 0.2 original   # GPT 0.1
        warmup_epochs: 0.05    # 0.01 original  # GPT 0.05
        lr_min_factor: 0.05    # 0.01 original  # GPT 0.05

    max_epochs: 5
    force_coefficient: 100
    energy_coefficient: 4
    clip_grad_norm: 100
    ema_decay: 0.999
    loss_position: mae
    loss_energy: mae
    loss_force: l2mae

    eval_every: 400          # 10000 (original val)
    position_coefficient: 1
    num_ctx_atoms: 0.1         # Context windows for the autoregressive task. It mimics the questions or context in LLM. if it is < 1, it means it is a percentage. 

    '''

    
    config.num_ctx_atoms = float(args.num_ctx_atoms) if hasattr(args, "num_ctx_atoms") else 0.1
    config.eval_on_free_atoms_only = False
    
    
    # Handle taskifying node-level keys
    config.mt_dataset = MTDatasetConfig(
        taskify_keys_graph=[
            "y", "y_scale", "force_scale", "positions_scale", "atomic_numbers_scale"
            ],
        taskify_keys_node=["force", "pos", "atomic_numbers", "center"],
        sample_type="temperature",
        sample_temperature=1.0,
        balanced=True,
        strict=True,
    )
    

    # Assign seed for config and pytorch lightning
    config.trainer.seed = args.seed
    config.set_seed(args.seed)
    
    # Assign validation check interval
    config.trainer.val_check_interval = args.val_interval # Run validation 5 times per epoch (every 20% of training)
    
    # Set the model trainer settings for maximum performance
    config.trainer.precision = "16-mixed"
    config.trainer.set_float32_matmul_precision = "medium"
    config.trainer.supports_parameter_hooks = False
    config.trainer.supports_skip_batch_exception = False


    # Epochs
    config.trainer.max_epochs = args.epochs

    # Batch size
    config.batch_size = args.batch_size
    # config.eval_batch_size = 20
    # config.eval_batch_size = 2

    weight_decay = 0.001
    warmup_epochs = 0.1  # 0.05 before (official 2M - 30 epochs: 0.1, official all: 0.01 )
    warmup_factor = 0.2  # 0.2 before (official 2M - 30 epochs: 0.2, official all: 0.2 )
    lr_min_factor = 0.01 # 0.05 before (official 2M - 30 epochs: 0.01, official all: 0.01 )

    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=args.lr,
        amsgrad=False,
        # betas=(0.9, 0.95),
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    
    # config.trainer.accumulate_grad_batches = 16
    config.trainer.optimizer.log_grad_norm = True
    config.trainer.optimizer.gradient_clipping = GradientClippingConfig(
        value=100, # set to 100 in the other repo
        algorithm="norm",
    )
    # LR Scheduler settings
    
    # num_gpus = torch.cuda.device_count()
    # n_iter_per_epoch = math.ceil(args.batch_size / num_gpus)
    # n_iter_per_epoch = n_iter_per_epoch // config.trainer.accumulate_grad_batches
    # warmup_steps = warmup_epochs * n_iter_per_epoch
    # config.lr_scheduler = WarmupCosineSchedulerConfig(
    #     warmup_steps=warmup_steps,
    #     warmup_start_lr_factor=warmup_factor,
    #     min_lr_factor=lr_min_factor,
    #     max_epochs=args.epochs, # usually 30 
    
    # )

    config.lr_scheduler = WarmupCosineLambdaLRSchedulerConfig(
        warmup_epochs=warmup_epochs,
        warmup_factor=warmup_factor,
        lr_min_factor=lr_min_factor,
        max_epochs=args.epochs,  # usually 30
    )

    # config.lr_scheduler = WarmupCosineLambdaLRSchedulerConfig(
    #     warmup_steps=5000,
    #     warmup_start_lr_factor=warmup_factor,
    #     lr_min_factor=lr_min_factor,
    #     max_epochs=args.epochs,  # usually 30
    # )

    # EMA settings
    config.ema = EMAConfig(decay=0.999)
    
    # Set backbone config
    if args.medium:
        config.backbone = EquiformerV2Config.medium()
    elif args.large:
        config.backbone = EquiformerV2Config.large()
    else:
        config.backbone = EquiformerV2Config.base()
        
        param_optimizer_settings = {
            "no_weight_decay": AdamWConfig(
                    lr=args.lr,
                    amsgrad=False,
                    betas=(0.9, 0.999),
                    weight_decay=0.0,
                ), 
        }
        config.parameter_specific_optimizers = pretrain_make_parameter_specific_optimizer_config_(
            config, 
            config.backbone.num_layers,
            param_optimizer_settings,
        )

    # override default settings to handle large atoms
    config.backbone.max_num_elements = 100

    # Set data config
    config.num_workers = args.num_workers

    config.primary_metric = PrimaryMetricConfig(name="loss", mode="min")
    config.ckpt_best = CheckpointBestConfig()


