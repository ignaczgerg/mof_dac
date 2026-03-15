"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
from pathlib import Path
import math
import numpy as np
from jmp.lightning import GradientClippingConfig

from ...models.equiformer_v2.config import (
    EquiformerV2Config, 
    AutoregressiveLlamaEquiformerV2Config
)
from ...tasks.config import AdamWConfig
from ...modules.ema import EMAConfig
from ...tasks.finetune import FinetuneConfigBase
from ...models.base import GraphModelMixin
from ...modules.dataset.dataset_transform import compute_dataset_stats
from ...modules.dataset.common import wrap_common_dataset
from ...tasks.finetune.base import (
    CheckpointBestConfig,
    ParamSpecificOptimizerConfig,
    RLPConfig, 
    WarmupCosRLPConfig
)
from ...tasks.finetune.model_wrapper_base import (
    EquiformerV2ModelWrapper, 
    AutoregLlamaEquiformerV2ModelWrapper
)
from ...utils.param_specific_util import (
    make_parameter_specific_optimizer_config, 
    make_parameter_specific_optimizer_config_from_optimizer_config, 
)
from ...tasks.finetune.adsorption_db import MOFFeatureConfig

BASE_PATH = Path(__file__).resolve().parent.parent.parent


def equiformer_v2_ft_config_(
    config: FinetuneConfigBase,
    ckpt_path: Path,
    ema_backbone: bool = True,
    disable_force_output_heads: bool = True,
    args: argparse.Namespace = argparse.Namespace(),
):

    # Assign the model wrapper
    config.model_cls = EquiformerV2ModelWrapper
    
    # Assign seed for config and pytorch lightning
    config.trainer.seed = args.seed
    config.set_seed(args.seed)

    # Report MAPE in addition to MAE
    config.metrics.report_mape = True
    config.metrics.report_smape = True

    # Set the model trainer settings for maximum performance
    config.trainer.precision = args.precision
    config.trainer.set_float32_matmul_precision = "high"
    config.trainer.supports_parameter_hooks = False
    config.trainer.supports_skip_batch_exception = False

    # Set backbone config
    if args.very_small_qm9:
        config.backbone = EquiformerV2Config.very_small_qm9()
    elif args.small:
        config.backbone = EquiformerV2Config.small()
    elif args.medium:
        config.backbone = EquiformerV2Config.medium()
    elif args.large:
        config.backbone = EquiformerV2Config.large()


    if "odac" in args.checkpoint_tag and args.small:
        config.backbone = EquiformerV2Config.small_odac()
    elif "odac" in args.checkpoint_tag and args.large:
        config.backbone = EquiformerV2Config.large_odac()

    # Inject MOF feature fusion settings into the backbone config
    config.backbone.mof_feature_fusion = MOFFeatureConfig(
        enabled=args.enable_feature_fusion,
        feature_name=args.fusion_feature_name,
        fusion_type=args.fusion_type,
        hidden_dim=args.fusion_hidden_dim,
    )

    # For our custom checkpoints, we need to change num_elements from 90 to 100 
    # if args.checkpoint_tag not in ["jmp", "oc20_public"]:
    #     config.backbone.max_num_elements = 100 
    # # ["jmp", "oc20_public"] expanded in setup_finetune.py instead
    config.backbone.max_num_elements = 100 

    '''
    LR Optim/Scheduler settings (copied from the original EquiformerV2)

    optim:
    batch_size: 4 
    eval_batch_size: 2
    grad_accumulation_steps: 64 
    # gradient accumulation: effective batch size = `grad_accumulation_steps` * `batch_size` * (num of GPUs)
    load_balancing: atoms
    num_workers: 8
    lr_initial: 0.0004    # [0.0002, 0.0004], eSCN uses 0.0008 for batch size 96

    optimizer: AdamW
    optimizer_params:
        weight_decay: 0.001
    scheduler: LambdaLR
    scheduler_params:
        lambda_type: cosine
        warmup_factor: 0.2
        warmup_epochs: 0.01
        lr_min_factor: 0.01         

    max_epochs: 3
    force_coefficient: 100
    energy_coefficient: 4
    clip_grad_norm: 100
    ema_decay: 0.999
    loss_position: mae
    loss_energy: mae
    loss_force: l2mae
    '''

    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=args.lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    config.trainer.optimizer.log_grad_norm = True
    config.trainer.optimizer.gradient_clipping = GradientClippingConfig(
        value=1.0,
        algorithm="value",
    )

   
    warmup_epochs= math.ceil(0.01 * args.epochs)
    warmup_start_lr_factor=2.0e-2
    min_lr_factor=0.01
    # if args.dataset_name == "qmof":
    #     warmup_epochs=0 
    #     warmup_start_lr_factor=1.0

    config.lr_scheduler = WarmupCosRLPConfig(
        warmup_epochs=warmup_epochs,
        warmup_start_lr_factor=warmup_start_lr_factor,
        should_restart=False,
        max_epochs=args.epochs,
        min_lr_factor=min_lr_factor,
        rlp=RLPConfig(patience=3, factor=0.8),
    )

    # TODO: Implement parameter specific optimizers for differnet sizes 
    if args.medium:
        raise NotImplementedError(
            "Only RLPConfig is supported if `parameter_specific_optimizers` is None."
            "Currently, only the small/large EquiformerV2 model are supported."
            )
    else:
        # max_lr_scales = {
        #     "embedding": 0.3,
        #     "blocks_0": 0.35,
        #     "blocks_1": 0.40,
        #     "blocks_2": 0.55,
        #     "blocks_3": 0.625,
        #     "blocks_4": 0.685,
        #     "blocks_5": 0.75,
        #     "blocks_6": 0.9,
        #     "blocks_7": 1.0,
        # }
        if config.backbone.num_layers == 5:
            max_lr_scales = {
                "embedding": 1.0,
                "blocks_0": 1.0,
                "blocks_1": 1.0,
                "blocks_2": 1.0,
                "blocks_3": 1.0,
                "blocks_4": 1.0,
            }
        elif config.backbone.num_layers == 8:
            max_lr_scales = {
                "embedding": 1.0,
                "blocks_0": 1.0,
                "blocks_1": 1.0,
                "blocks_2": 1.0,
                "blocks_3": 1.0,
                "blocks_4": 1.0,
                "blocks_5": 1.0,
                "blocks_6": 1.0,
                "blocks_7": 1.0,
            }
        elif config.backbone.num_layers == 20:
            max_lr_scales = {
                "embedding": 1.0,
                "blocks_0": 1.0,
                "blocks_1": 1.0,
                "blocks_2": 1.0,
                "blocks_3": 1.0,
                "blocks_4": 1.0,
                "blocks_5": 1.0,
                "blocks_6": 1.0,
                "blocks_7": 1.0,
                "blocks_8": 1.0,
                "blocks_9": 1.0,
                "blocks_10": 1.0,
                "blocks_11": 1.0,
                "blocks_12": 1.0,
                "blocks_13": 1.0,
                "blocks_14": 1.0,
                "blocks_15": 1.0,
                "blocks_16": 1.0,
                "blocks_17": 1.0,
                "blocks_18": 1.0,
                "blocks_19": 1.0,
            }

        config.parameter_specific_optimizers = make_parameter_specific_optimizer_config(
            config,
            config.backbone.num_layers, 
            max_lr_scales,
        )

    # Checkpoint loading settings
    # We want to use EMA weights from pretraining
    config.meta["ckpt_path"] = ckpt_path
    config.meta["ema_backbone"] = ema_backbone

    if args.disable_ema:
        print("[INFO] Disabling EMA...")
        config.ema = None
    else:
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
    
    print("[INFO] Finished setting up EquiformerV2 finetune config.")
    _maybe_set_dataset_stats(config, args)
    print("average num_nodes:", getattr(config.backbone, "avg_num_nodes", None))
    print("average degree:", getattr(config.backbone, "avg_degree", None))
    print("[INFO] Finished computing dataset stats (if enabled).")

def _build_one_dataset_for_stats(config, split: str = "train"):
    tasks = getattr(config, "train_tasks", None) or getattr(config, "tasks", [])
    for task in tasks:
        cfg = getattr(task, f"{'train' if split=='train' else 'val'}_dataset", None)
        if cfg is None:
            continue
        try:
            ds = cfg.create_dataset(split)
            ds = wrap_common_dataset(ds, cfg) 
            return ds
        except Exception:
            continue
    return None

class _MiniGraph(GraphModelMixin):
    def __init__(self, *, cutoff: float, max_neighbors: int, use_pbc: bool,
                 use_pbc_single: bool = False, enforce_max_neighbors_strictly: bool = True):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.otf_graph = True
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly


def _maybe_set_dataset_stats(config, args):
    try:
        if getattr(config, "backbone", None) is None:
            raise ValueError("No backbone in config.")
            
        ds = _build_one_dataset_for_stats(config, split="train") or _build_one_dataset_for_stats(config, split="val")
        if ds is None or len(ds) == 0:
            return

        max_neighbors = getattr(config.backbone, "max_neighbors", 48)
        use_pbc = not getattr(args, "no_pbc", False)
        cutoff = getattr(args, "cutoff", getattr(config.backbone, "max_radius", 5.0))

        mini = _MiniGraph(
            cutoff=float(cutoff),
            max_neighbors=int(max_neighbors),
            use_pbc=bool(use_pbc),
            use_pbc_single=False,
            enforce_max_neighbors_strictly=True,
        )

        stats = compute_dataset_stats(
            ds,
            sample_size=int(getattr(args, "avg_stats_sample_size", 1000)),
            compute_degree=True,
            generate_graph=mini.generate_graph,
            generate_graph_kwargs=dict(
                cutoff=float(cutoff),
                max_neighbors=int(max_neighbors),
                use_pbc=bool(use_pbc),
                otf_graph=True,
                enforce_max_neighbors_strictly=True,
                use_pbc_single=False,
            ),
            ensure_batch=True,
            require_cell_if_pbc=True,
            force_otf=True,
            pbar=False,
        )

        avg_nodes = float(stats.get("avg_num_nodes", np.nan))
        avg_degree = float(stats.get("avg_degree", np.nan))
        if np.isfinite(avg_nodes):
            config.backbone.avg_num_nodes = avg_nodes
            print(f"[INFO] Set backbone.avg_num_nodes = {avg_nodes:.2f} from dataset stats.")
        else:
            print("[WARNING] Could not compute avg_num_nodes from dataset stats.")
        if np.isfinite(avg_degree):
            config.backbone.avg_degree = avg_degree
            print(f"[INFO] Set backbone.avg_degree = {avg_degree:.2f} from dataset stats.")
        else:
            print("[WARNING] Could not compute avg_degree from dataset stats.")

    except Exception:
        print("[WARNING] Could not compute dataset stats for avg_num_nodes and avg_degree.")
        pass





def autoreg_llama_equiformer_v2_ft_config_(
    config: FinetuneConfigBase,
    ckpt_path: Path,
    ema_backbone: bool = True,
    disable_force_output_heads: bool = True,
    args: argparse.Namespace = argparse.Namespace(),
):

    # Assign the model wrapper
    config.model_cls = AutoregLlamaEquiformerV2ModelWrapper
    
    # Assign seed for config and pytorch lightning
    config.trainer.seed = args.seed
    config.set_seed(args.seed)

    # Report MAPE in addition to MAE
    config.metrics.report_mape = True
    config.metrics.report_smape = True

    # Set the model trainer settings for maximum performance
    config.trainer.precision = args.precision
    config.trainer.set_float32_matmul_precision = "high"
    config.trainer.supports_parameter_hooks = False
    config.trainer.supports_skip_batch_exception = False

    # Set backbone config
    # if args.medium:
    #     config.backbone = EquiformerV2Config.medium()
    # elif args.large:
    #     config.backbone = EquiformerV2Config.large()
    # else:
    #     config.backbone = EquiformerV2Config.base()
    config.backbone = AutoregressiveLlamaEquiformerV2Config.base()

    
    config.backbone.use_position_enc = False
    
    '''
    LR Optim/Scheduler settings (copied from the original EquiformerV2)

    optim:
    batch_size: 4 
    eval_batch_size: 2
    grad_accumulation_steps: 64 
    # gradient accumulation: effective batch size = `grad_accumulation_steps` * `batch_size` * (num of GPUs)
    load_balancing: atoms
    num_workers: 8
    lr_initial: 0.0004    # [0.0002, 0.0004], eSCN uses 0.0008 for batch size 96

    optimizer: AdamW
    optimizer_params:
        weight_decay: 0.001
    scheduler: LambdaLR
    scheduler_params:
        lambda_type: cosine
        warmup_factor: 0.2
        warmup_epochs: 0.01
        lr_min_factor: 0.01         

    max_epochs: 3
    force_coefficient: 100
    energy_coefficient: 4
    clip_grad_norm: 100
    ema_decay: 0.999
    loss_position: mae
    loss_energy: mae
    loss_force: l2mae
    '''

    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=0.0002,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.001, # this is changed to match the config
    )
    config.trainer.optimizer.log_grad_norm = True
    config.trainer.optimizer.gradient_clipping = GradientClippingConfig(
        value=1.0,
        algorithm="value",
    )

   
    warmup_epochs=1
    warmup_start_lr_factor=0.2
    min_lr_factor=0.01

    config.lr_scheduler = WarmupCosRLPConfig(
        warmup_epochs=warmup_epochs,
        warmup_start_lr_factor=warmup_start_lr_factor,
        should_restart=False,
        max_epochs=args.epochs,
        min_lr_factor=min_lr_factor,
        rlp=RLPConfig(patience=3, factor=0.8),
    )

    # TODO: Implement parameter specific optimizers for differnet sizes 
    if args.medium or args.large:
        raise NotImplementedError(
            "Only RLPConfig is supported if `parameter_specific_optimizers` is None."
            "Currently, we only the small EquiformerV2 model is supported."
            )
    else:
        # max_lr_scales = {
        #     "embedding": 0.3,
        #     "layer_0": 0.329,
        #     "layer_1": 0.358,
        #     "layer_2": 0.387,
        #     "layer_3": 0.417,
        #     "layer_4": 0.446,
        #     "layer_5": 0.475,
        #     "layer_6": 0.504,
        #     "layer_7": 0.533,
        #     "layer_8": 0.562,
        #     "layer_9": 0.592,
        #     "layer_10": 0.621,
        #     "layer_11": 0.65,
        #     "layer_12": 0.679,
        #     "layer_13": 0.708,
        #     "layer_14": 0.737,
        #     "layer_15": 0.767,
        #     "layer_16": 0.796,
        #     "layer_17": 0.825,
        #     "layer_18": 0.854,
        #     "layer_19": 0.883,
        #     "layer_20": 0.912,
        #     "layer_21": 0.942,
        #     "layer_22": 0.971,
        #     "layer_23": 1.0,
        #     "out_mlp": 1.0,
        # }

        # config.parameter_specific_optimizers = make_parameter_specific_optimizer_config(
        #     config,
        #     24, # default num of layers in HuggingFaceTB/SmolLM2-1.7B
        #     max_lr_scales,
        # )
        
        param_optimizer_settings = {
            "no_weight_decay": AdamWConfig(
                    lr=0.0002,
                    amsgrad=False,
                    betas=(0.9, 0.95),
                    weight_decay=0.0,
                ), 
            "llm_no_weight_decay": AdamWConfig(
                    lr=0.00001,
                    amsgrad=False,
                    betas=(0.9, 0.95),
                    weight_decay=0.0,
                ),
        }
    
    config.parameter_specific_optimizers = make_parameter_specific_optimizer_config_from_optimizer_config(
        config, 
        24,  # default num of layers in HuggingFaceTB/SmolLM2-1.7B
        param_optimizer_settings,
    )

    # Checkpoint loading settings
    # We want to use EMA weights from pretraining
    config.meta["ckpt_path"] = ckpt_path
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
