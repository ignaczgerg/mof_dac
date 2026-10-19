"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
from typing import Any, cast
from collections.abc import Callable

from jmp.tasks.finetune.base import (
    FinetuneConfigBase,
    ParamSpecificOptimizerConfig,
    WarmupCosRLPConfig,
)
from jmp.tasks.pretrain.module import (
    PretrainConfig,
    ParamSpecificOptimizerConfig as PretrainParamSpecificOptimizerConfig,
    WarmupCosineSchedulerConfig, 
    WarmupCosineLambdaLRSchedulerConfig,
)
from jmp.tasks.config import AdamWConfig
import jmp.tasks.finetune.model_wrapper_base as models
import jmp.tasks.pretrain.model_wrapper_base as pretrain_models
from typing_extensions import TypeVar


def GEMNET_PARAMETER_PATTERNS(num_blocks: int):
    return {
        "embedding": ["embedding.*"],
        "additional_embedding": ["additional_embedding.*"],
        "bases": ["backbone.bases.*"],
        # "all_int_blocks": ["backbone.int_blocks.*"],
        **{
            f"int_blocks_{i}": [f"backbone.int_blocks.{i}.*"] for i in range(num_blocks)
        },
        # "all_out_blocks": ["backbone.out_blocks.*"],
        **{
            f"out_blocks_{i}": [f"backbone.out_blocks.{i}.*"]
            for i in range(num_blocks + 1)
        },
        **{
            f"blocks_{i}": [
                f"backbone.int_blocks.{i}.*",
                f"backbone.out_blocks.{i+1}.*",
                *(["backbone.out_blocks.0.*"] if i == 0 else []),
            ]
            for i in range(num_blocks)
        },
        "out_mlp_E": ["backbone.out_mlp.E.*"],
    }


def EQUIFORMER_V2_PARAMETER_PATTERNS(num_blocks: int):
    return {
        "embedding": [
            "backbone.sphere_embedding.*",
            "backbone.SO3_rotation.*",
            "backbone.mapping_reduced.*",
            "backbone.SO3_grid.*",
            "backbone.edge_degree_embedding.*",
            ],

        **{
            f"blocks_{i}": [
                f"backbone.blocks.{i}.*",
            ] for i in range(num_blocks)
        },

        "out_mlp": ["backbone.norm.*"],
        
        "no_weight_decay": [
            # in models/equiformer_v2/trainer/base_trainer_v2.py
            "*.bias",
            "*.affine_weight",
            "*.affine_bias",
            "*.mean_shift",
            "*bias.*",
            # in models/equiformer_v2/equiformer_v2.py
            "backbone.norm.*",
        ],
    }

def AUTOREG_LLAMA_EQUIFORMER_V2_PARAMETER_PATTERNS(num_blocks: int):
    return {
        "embedding": [
            "backbone.sphere_embedding.*",
            "backbone.SO3_rotation.*",
            "backbone.mapping_reduced.*",
            "backbone.SO3_grid.*",
            "backbone.edge_degree_embedding.*",
            ],

        **{
            f"layer_{i}": [
                f"backbone.model.model.layers.{i}.*",
            ] for i in range(num_blocks)
        },

        "out_mlp": [
            "backbone.model.model.norm.*",
            "backbone.model.model.lm_head.*",
            "backbone.proj.*",
            "backbone.out_proj.*",
            "backbone.LLMfromS03Embedding.*",
            "backbone.norm.*", 
            ],
        
        #no weight decay  
        "no_weight_decay": [
            # in models/equiformer_v2/trainer/base_trainer_v2.py
            "*.bias",
            "*.affine_weight",
            "*.affine_bias",
            "*.mean_shift",
            "*bias.*",
            # in models/equiformer_v2/equiformer_v2.py
            "backbone.norm.*",
        ],
                
        "llm_no_weight_decay": [
            # in models/equiformer_v2/equiformer_v2.py
            "backbone.model.model.*",
        ],
    }


TConfig = TypeVar("TConfig", infer_variance=True)

def select_model_parameter_patterns(
    config: FinetuneConfigBase | PretrainConfig,
    ) -> Callable[[int], dict[str, list[str]]]:
    # TODO: Support other model classes
    match config.model_cls:
        case models.GemNetModelWrapper | pretrain_models.GemNetModelWrapper:
            PARAMETER_PATTERNS = GEMNET_PARAMETER_PATTERNS
        case (models.EquiformerV2ModelWrapper | pretrain_models.EquiformerV2ModelWrapper |
              pretrain_models.AutoregressiveEquiformerV2ModelWrapper):
            PARAMETER_PATTERNS = EQUIFORMER_V2_PARAMETER_PATTERNS
        case (models.AutoregLlamaEquiformerV2ModelWrapper |
              pretrain_models.AutoregLlamaEquiformerV2ModelWrapper):
            PARAMETER_PATTERNS = AUTOREG_LLAMA_EQUIFORMER_V2_PARAMETER_PATTERNS
        case _:
            raise ValueError(
                f"Unsupported model class: {config.model_cls} for parameter_specific_optimizer." 
                "Supported classes are GemNetModelWrapper and EquiformerV2ModelWrapper."
            )
    return PARAMETER_PATTERNS


def make_parameter_specific_optimizer_config(
    config: FinetuneConfigBase,
    num_blocks: int,
    max_lr_scales: dict[str, float],
):
    PARAMETER_PATTERNS = select_model_parameter_patterns(config)
    base_lr = config.optimizer.lr

    parameter_specific_optimizers: list[ParamSpecificOptimizerConfig] = []
    max_lr_scales = cast(dict[str, Any], max_lr_scales)
    for name, lr_scale in max_lr_scales.items():
        assert isinstance(lr_scale, float), f"max_lr_scales[{name}] must be float"

        optimizer = copy.deepcopy(config.optimizer)
        optimizer.lr = base_lr * lr_scale

        lrs = None
        match config.lr_scheduler:
            case WarmupCosRLPConfig():
                lrs = copy.deepcopy(config.lr_scheduler)
                # We now scale down the cos annealing min LR factor
                #   so that the final LR is the same as the original config.
                lrs.min_lr_factor = lrs.min_lr_factor / lr_scale
                lrs.min_lr_factor = max(0.01, min(0.99, lrs.min_lr_factor))
            case _:
                raise ValueError(
                    "You must set config.lr_scheduler to WarmupCosRLPConfig to use parameter specific optimizers."
                )

        assert (
            (parameter_patterns := PARAMETER_PATTERNS(num_blocks).get(name)) is not None
        ), f"PARAMETER_PATTERNS[{name}] is None. You must set PARAMETER_PATTERNS[{name}]"
        parameter_specific_optimizers.append(
            ParamSpecificOptimizerConfig(
                paremeter_patterns=parameter_patterns,
                optimizer=optimizer,
                lr_scheduler=lrs,
            )
        )

    return parameter_specific_optimizers


def make_parameter_specific_optimizer_config_from_optimizer_config(
    config: FinetuneConfigBase,
    num_blocks: int,
    param_optimizer_settings: dict[str, AdamWConfig],
):
    PARAMETER_PATTERNS = select_model_parameter_patterns(config)
    base_lr = config.optimizer.lr

    parameter_specific_optimizers: list[ParamSpecificOptimizerConfig] = []
    param_optimizer_settings = cast(dict[str, Any], param_optimizer_settings)
    for name, optimizer in param_optimizer_settings.items():
        assert isinstance(optimizer, AdamWConfig), f"max_lr_scales[{name}] must be AdamWConfig"

        lrs = None
        match config.lr_scheduler:
            case WarmupCosRLPConfig():
                lrs = copy.deepcopy(config.lr_scheduler)
                lr_scale = optimizer.lr / base_lr
                # We now scale down the cos annealing min LR factor
                #   so that the final LR is the same as the original config.
                lrs.min_lr_factor = lrs.min_lr_factor / lr_scale
                lrs.min_lr_factor = max(0.01, min(0.99, lrs.min_lr_factor))
            case _:
                raise ValueError(
                    "You must set config.lr_scheduler to WarmupCosRLPConfig to use parameter specific optimizers."
                )

        assert (
            (parameter_patterns := PARAMETER_PATTERNS(num_blocks).get(name)) is not None
        ), f"PARAMETER_PATTERNS[{name}] is None. You must set PARAMETER_PATTERNS[{name}]"
        parameter_specific_optimizers.append(
            ParamSpecificOptimizerConfig(
                paremeter_patterns=parameter_patterns,
                optimizer=optimizer,
                lr_scheduler=lrs,
            )
        )

    return parameter_specific_optimizers

def pretrain_make_parameter_specific_optimizer_config_(
    config: PretrainConfig,
    num_blocks: int,
    param_optimizer_settings: dict[str, AdamWConfig],
):
    PARAMETER_PATTERNS = select_model_parameter_patterns(config)
    base_lr = config.optimizer.lr

    parameter_specific_optimizers: list[PretrainParamSpecificOptimizerConfig] = []
    param_optimizer_settings = cast(dict[str, Any], param_optimizer_settings)
    for name, optimizer in param_optimizer_settings.items():
        assert isinstance(optimizer, AdamWConfig), f"max_lr_scales[{name}] must be AdamWConfig"

        lrs = None
        match config.lr_scheduler:
            case WarmupCosineSchedulerConfig():
                lrs = copy.deepcopy(config.lr_scheduler)
                lr_scale = optimizer.lr / base_lr
                # We now scale down the cos annealing min LR factor
                #   so that the final LR is the same as the original config.
                lrs.min_lr_factor = lrs.min_lr_factor / lr_scale
                lrs.min_lr_factor = max(0.01, min(0.99, lrs.min_lr_factor))

            case WarmupCosineLambdaLRSchedulerConfig():
                lrs = copy.deepcopy(config.lr_scheduler)
                lr_scale = optimizer.lr / base_lr
                # We now scale down the cos annealing min LR factor
                #   so that the final LR is the same as the original config.
                lrs.lr_min_factor = lrs.lr_min_factor / lr_scale
                lrs.lr_min_factor = max(0.01, min(0.99, lrs.lr_min_factor))
            case _:
                raise ValueError(
                    "You must set config.lr_scheduler to WarmupCosineSchedulerConfig to use parameter specific optimizers."
                )

        assert (
            (parameter_patterns := PARAMETER_PATTERNS(num_blocks).get(name)) is not None
        ), f"PARAMETER_PATTERNS[{name}] is None. You must set PARAMETER_PATTERNS[{name}]"
        parameter_specific_optimizers.append(
            PretrainParamSpecificOptimizerConfig(
                paremeter_patterns=parameter_patterns,
                optimizer=optimizer,
                lr_scheduler=lrs,
            )
        )

    return parameter_specific_optimizers