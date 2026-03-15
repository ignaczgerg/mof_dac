"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from pathlib import Path
from typing import cast 
import argparse
import math 
import copy
from typing import Sequence, Literal

from ...utils.env_utils import load_env_paths
from ...modules.transforms.normalize import NormalizationConfig as NC
from ...modules.dataset.concat_dataset import MTDatasetConfig
from ...modules.dataset.common import DatasetSampleNConfig
from ...tasks.config import AdamWConfig
from ...tasks.finetune import AdsorptionDbConfig 
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import PrimaryMetricConfig
from ...tasks.finetune.adsorption_db import (
    AdsorptionDbTarget, TaskConfig, FinetuneLmdbDatasetConfig, MOFFeatureConfig
    )

from ...tasks.finetune.model_wrapper_base import MulticlassClassificationTargetConfig

# These stats are just for refernce. The default behaviour is 
# on-the-fly normalization
STATS: dict[str, NC] = {
"qst_co2": NC(mean=-26.07204216, std=6.82141212, normalization_type="standard"),
"qst_h2o": NC(mean=-35.81842328, std=13.81936149, normalization_type="standard"),
"kh_co2": NC(mean=0.00020583, std=0.00207071, normalization_type="standard"),
"kh_h2o": NC(mean=0.02502187, std=0.10112895, normalization_type="standard"),
"selectivity_co2_h2o": NC(mean=3.73860177, std=20.20365806, normalization_type="standard"),
"co2_uptake": NC(mean=0.00729278, std=0.04347385, normalization_type="standard"),
"qst_n2":             NC(mean=-1.440007e+01, std=3.911949e+00, normalization_type="standard" ),
"kh_n2":              NC(mean=3.242458e-06, std=2.810460e-06 , normalization_type="standard"),
"selectivity_co2_n2": NC(mean=1.020979e+03, std=4.275029e+04 , normalization_type="standard"),
"n2_uptake":          NC(mean=1.303642e-04, std=1.110940e-04 , normalization_type="standard"),
}


LOG_STATS: dict[str, NC] = {
"qst_co2": NC(mean=3.22356297, std=0.28249086, normalization_type="log"),
"qst_h2o": NC(mean=3.49313804, std=0.43487195, normalization_type="log"),
"kh_co2": NC(mean=-9.79689810, std=1.38127887, normalization_type="log"),
"kh_h2o": NC(mean=-8.42486847, std=3.30167592, normalization_type="log"),
"selectivity_co2_h2o": NC(mean=-1.37203222, std=2.98345944, normalization_type="log"),
"co2_uptake": NC(mean=-6.10493189, std=1.38102965, normalization_type="log"),
"qst_n2": NC(mean=-14.19377044, std=3.80194344, normalization_type="log"),
"kh_n2": NC(mean=-12.83952345, std=0.66957234, normalization_type="log"),
"selectivity_co2_n2": NC(mean=3.05349598, std=1.05021461, normalization_type="log"),
"n2_uptake": NC(mean=-9.14759207, std=0.67391268, normalization_type="log"),
}

STATS_DB2: dict[str, NC] = {
"qst_co2": NC(mean=-21.329700577143377, std=5.267456123165425, normalization_type="standard"),
"qst_h2o": NC(mean=-35.579342134543474, std=10.501804952100375, normalization_type="standard"),
"kh_co2": NC(mean=7.463583489022185e-05, std=0.0021739318346768667, normalization_type="standard"),
"kh_h2o": NC(mean=0.012248716692478429, std=0.060077771745325366, normalization_type="standard"),
"selectivity_co2_h2o": NC(mean=1.8437359571885585, std=112.30922150928325, normalization_type="standard"),
"co2_uptake": NC(mean=0.002658264774626291, std=0.013859310237578045, normalization_type="standard"),
}

LOG_STATS_DB2: dict[str, NC] = {
"qst_co2": NC(mean=-21.329700577143377, std=5.267456123165425, normalization_type="standard"),
"qst_h2o": NC(mean=-35.579342134543474, std=10.501804952100375, normalization_type="standard"),
"kh_co2": NC(mean=-10.40316590478883, std=0.9921236995164925, normalization_type="log"),
"kh_h2o": NC(mean=-8.103607446290459, std=2.7491095788897586, normalization_type="log"),
"selectivity_co2_h2o": NC(mean=-2.299558465085649, std=2.729825618704778, normalization_type="log"),
"co2_uptake": NC(mean=-6.7006051469045635, std=1.0064174393812682, normalization_type="log"),
}


STATS_COF2: dict[str, NC] = {
"qst_co2": NC(mean=-21.329700577143377, std=5.267456123165425, normalization_type="standard"),
"qst_h2o": NC(mean=-35.579342134543474, std=10.501804952100375, normalization_type="standard"),
"kh_co2": NC(mean=7.463583489022185e-05, std=0.0021739318346768667, normalization_type="standard"),
"kh_h2o": NC(mean=0.012248716692478429, std=0.060077771745325366, normalization_type="standard"),
"selectivity_co2_h2o": NC(mean=1.8437359571885585, std=112.30922150928325, normalization_type="standard"),
"co2_uptake": NC(mean=0.002658264774626291, std=0.013859310237578045, normalization_type="standard"),
}

LOG_STATS_COF2: dict[str, NC] = {
"qst_co2": NC(mean=-21.329700577143377, std=5.267456123165425, normalization_type="standard"),
"qst_h2o": NC(mean=-35.579342134543474, std=10.501804952100375, normalization_type="standard"),
"kh_co2": NC(mean=-10.40316590478883, std=0.9921236995164925, normalization_type="log"),
"kh_h2o": NC(mean=-8.103607446290459, std=2.7491095788897586, normalization_type="log"),
"selectivity_co2_h2o": NC(mean=-2.299558465085649, std=2.729825618704778, normalization_type="log"),
"co2_uptake": NC(mean=-6.7006051469045635, std=1.0064174393812682, normalization_type="log"),
}

def _build_normalization_map(
    dataset_key: str,
    targets: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, NC]:
    """
    attempt to map the normalization for the specific dataset and targets
    in multihead
    """
    stats_source = {
        "adsorption_db1_merged": {"standard": STATS,      "log": LOG_STATS},
        "adsorption_db2_merged": {"standard": STATS_DB2,  "log": LOG_STATS_DB2},
        "adsorption_cof_db2": {"standard": STATS_COF2,  "log": LOG_STATS_COF2},
    }
    if dataset_key not in stats_source:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")

    def _norm_type_for(i: int) -> str:
        nt = getattr(args, "normalization_type", "standard")
        if isinstance(nt, (list, tuple)):
            return nt[i]
        return nt

    has_mu = isinstance(getattr(args, "norm_mean", None), (list, tuple))
    has_sd = isinstance(getattr(args, "norm_std", None), (list, tuple))

    norm_map: dict[str, NC] = {}
    for i, t in enumerate(targets):
        ntype = _norm_type_for(i)
        ntype = "log" if ntype == "log" else "standard"

        src_dict = stats_source[dataset_key][ntype]
        if t not in src_dict:
            raise KeyError(
                f"Normalization stats for target '{t}' not found in {dataset_key} ({ntype})."
            )

        nc = copy.deepcopy(src_dict[t])

        if has_mu and i < len(args.norm_mean):
            nc.mean = args.norm_mean[i]
        if has_sd and i < len(args.norm_std):
            nc.std = args.norm_std[i]

        norm_map[t] = nc

    return norm_map


def _apply_graph_scalar_reduction_to_config(
    config: AdsorptionDbConfig,
    targets: Sequence[str],
    args: argparse.Namespace,
):
    """Map args.graph_scalar_reduction onto the active targets separately"""

    reds = getattr(args, "graph_scalar_reduction", "mean")
    if isinstance(reds, str):
        reds = [reds]
    if len(reds) == 1:
        reds = reds * len(targets)
    assert len(reds) == len(targets), (
        f"--graph_scalar_reduction must be length 1 or {len(targets)} "
        f"(got {len(reds)})"
    )
    config.graph_scalar_reduction = {t: r for t, r in zip(targets, reds)}


def jmp_l_adsorption_db_config_(
        config: AdsorptionDbConfig,
        targets: list[AdsorptionDbTarget],
        base_path: Path,
        args: argparse.Namespace = argparse.Namespace(),
        # lr_base: float = 8.0e-5
    ):
    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=args.lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    # Assign key file(s) path
    config.meta["train_keys_file"] = base_path / "split_keys" / "train_keys.pkl"
    config.meta["val_keys_file"] = base_path / "split_keys" / "val_keys.pkl"
    config.meta["test_keys_file"] = base_path / "split_keys" / "test_keys.pkl"
    
    # Regularization settings
    if args.edge_dropout:
        config.backbone.edge_dropout = args.edge_dropout
    if args.dropout:
        config.backbone.dropout = args.dropout

    # Set data config
    config.batch_size = args.batch_size
    # config.epochs were disabled here because it was messing up the 
    # evaluate script. 
    # ideally, using config.epochs is fine, but I leave a TODO here.
    # config.epochs = args.epochs
    config.trainer.max_epochs = args.epochs
    # Set up dataset
    config.train_dataset = DC.adsorption_db_config(base_path, "train", args=args)
    config.val_dataset = DC.adsorption_db_config(base_path, "val", args=args)
    config.test_dataset = DC.adsorption_db_config(base_path, "test", args=args)

    graph_scalar_loss_coefficients: dict[AdsorptionDbTarget, float] = {}
    if isinstance(args.norm_mean, list):
        assert len(args.norm_mean) == len(args.norm_std), "Mean and std should have the same length"
        assert len(args.norm_mean) <= len(targets), "if custom mean/std is enabled, ensure targets are ordered"
    
    for idx, target in enumerate(targets):

        # Set up normalization
        stats_map = LOG_STATS if args.normalization_type[idx] == "log" else STATS
        if (normalization_config := stats_map.get(target)) is None:
            raise ValueError(
                f"{args.normalization_type.capitalize()} normalization for {target} not found"
                )
        try:
            normalization_config.mean = args.norm_mean[idx]
            normalization_config.std = args.norm_std[idx]
        except:
            pass
        config.normalization[target] = normalization_config

        # Set up loss coefficients
        try:
            graph_scalar_loss_coefficients[target] = args.targets_loss_coefficients[idx]
        except:
            graph_scalar_loss_coefficients[target] = config.graph_scalar_loss_coefficient_default
            
        

    # AdsorptionDb specific settings
    config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_mae", mode="min")


    # Make sure we only optimize for the target
    config.graph_scalar_targets = cast(list[str], targets)
    config.graph_scalar_loss_coefficients = cast(
        dict[str, float], graph_scalar_loss_coefficients)



def escaip_small_adsorption_db_config_(
        config: AdsorptionDbConfig,
        targets: list[AdsorptionDbTarget],
        base_path: Path,
        args: argparse.Namespace = argparse.Namespace(),
        # lr_base: float = 8.0e-5
    ):
    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=args.lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Assign key file(s) path
    config.meta["train_keys_file"] = base_path / "split_keys" / "train_keys.pkl"
    config.meta["val_keys_file"] = base_path / "split_keys" / "val_keys.pkl"
    config.meta["test_keys_file"] = base_path / "split_keys" / "test_keys.pkl"

    # Set data config
    config.batch_size = args.batch_size
    # TODO: support varying sizes of EScAIP
    # assert config.batch_size in [8, 16, 12], f"batch size {config.batch_size} not supported"
    # assert config.batch_size == 16, f"batch size {config.batch_size} has to be 16 for small EScAIP"
    
    # Set the backbone config
    config.backbone.batch_size = args.batch_size
    config.backbone.max_radius = args.cutoff #default is 8.0 here
    config.backbone.use_pbc = not args.no_pbc
    config.backbone.max_num_elements = 100
    
    '''
    pad_size = max_num_nodes_per_batch * batch_size - num_nodes
    -> max_num_nodes_per_batch >= num_nodes / batch_size for pad_size >= 0
    '''
    config.backbone.max_num_nodes_per_batch = math.ceil(
        args.max_natoms / args.batch_size if args.max_natoms else 1734
    ) # note max_natoms has to match in all the dataset splits
    
    # config.backbone.max_num_nodes_per_batch = 1680
    # for batch_size 3
    config.backbone.max_num_nodes_per_batch = 1734 # since max natoms is 5200
    
    
    # Set up dataset
    config.train_dataset = DC.adsorption_db_config(base_path, "train", args=args)
    config.val_dataset = DC.adsorption_db_config(base_path, "val", args=args)
    config.test_dataset = DC.adsorption_db_config(base_path, "test", args=args)

    graph_scalar_loss_coefficients: dict[AdsorptionDbTarget, float] = {}
    if isinstance(args.norm_mean, list):
        assert len(args.norm_mean) == len(args.norm_std), "Mean and std should have the same length"
        assert len(args.norm_mean) <= len(targets), "if custom mean/std is enabled, ensure targets are ordered"
    
    for idx, target in enumerate(targets):

        # Set up normalization
        stats_map = LOG_STATS if args.normalization_type[idx] == "log" else STATS
        if (normalization_config := stats_map.get(target)) is None:
            raise ValueError(
                f"{args.normalization_type.capitalize()} normalization for {target} not found"
                )
        try:
            normalization_config.mean = args.norm_mean[idx]
            normalization_config.std = args.norm_std[idx]
        except:
            pass
        config.normalization[target] = normalization_config

        # Set up loss coefficients
        try:
            graph_scalar_loss_coefficients[target] = args.targets_loss_coefficients[idx]
        except:
            graph_scalar_loss_coefficients[target] = config.graph_scalar_loss_coefficient_default
            
        

    # AdsorptionDb specific settings
    config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_mae", mode="min")


    # Make sure we only optimize for the target
    config.graph_scalar_targets = cast(list[str], targets)
    config.graph_scalar_loss_coefficients = cast(
        dict[str, float], graph_scalar_loss_coefficients)
    

def equiformer_v2_adsorption_db_config_(
        config: AdsorptionDbConfig,
        targets: list[AdsorptionDbTarget],
        base_path: Path,
        args: argparse.Namespace = argparse.Namespace(),
        # lr_base: float = 8.0e-5
    ):
    # Optimizer settings
    # config.optimizer = AdamWConfig(
    #     lr=args.lr,
    #     amsgrad=False,
    #     betas=(0.9, 0.95),
    #     weight_decay=args.weight_decay,
    # )

    # Assign key file(s) path
    config.meta["train_keys_file"] = base_path / "split_keys" / "train_keys.pkl"
    config.meta["val_keys_file"] = base_path / "split_keys" / "val_keys.pkl"
    config.meta["test_keys_file"] = base_path / "split_keys" / "test_keys.pkl"

    # Set data config
    config.batch_size = args.batch_size


    # Set max_num_elements 
    config.backbone.max_num_elements = 100
    # if args.checkpoint_tag == "odac_public" or "autoreg" in args.checkpoint_tag:
    #     config.backbone.max_num_elements = 100
        
    # Set up dataset
    config.train_dataset = DC.adsorption_db_config(base_path, "train", args=args)
    config.val_dataset = DC.adsorption_db_config(base_path, "val", args=args)
    config.test_dataset = DC.adsorption_db_config(base_path, "test", args=args)

    graph_scalar_loss_coefficients: dict[AdsorptionDbTarget, float] = {}
    if isinstance(args.norm_mean, list):
        assert len(args.norm_mean) == len(args.norm_std), "Mean and std should have the same length"
        assert len(args.norm_mean) <= len(targets), "if custom mean/std is enabled, ensure targets are ordered"
    
    for idx, target in enumerate(targets):

        # Set up normalization
        stats_map = LOG_STATS if args.normalization_type[idx] == "log" else STATS
        if (normalization_config := stats_map.get(target)) is None:
            raise ValueError(
                f"{args.normalization_type.capitalize()} normalization for {target} not found"
                )
        try:
            normalization_config.mean = args.norm_mean[idx]
            normalization_config.std = args.norm_std[idx]
        except:
            pass
        config.normalization[target] = normalization_config

        # Set up loss coefficients
        try:
            graph_scalar_loss_coefficients[target] = args.targets_loss_coefficients[idx]
        except:
            graph_scalar_loss_coefficients[target] = config.graph_scalar_loss_coefficient_default
            
        

    # AdsorptionDb specific settings
    config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_mae", mode="min")


    # Make sure we only optimize for the target
    config.graph_scalar_targets = cast(list[str], targets)
    config.graph_scalar_loss_coefficients = cast(
        dict[str, float], graph_scalar_loss_coefficients)
    

def jmp_l_adsorption_db_config_adabin_(
        config: AdsorptionDbConfig,
        targets: list[AdsorptionDbTarget],
        base_path: Path,
        args: argparse.Namespace = argparse.Namespace(),
        # lr_base: float = 8.0e-5
    ):
    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=args.lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    # Assign key file(s) path
    config.meta["train_keys_file"] = base_path / "split_keys" / "train_keys.pkl"
    config.meta["val_keys_file"] = base_path / "split_keys" / "val_keys.pkl"
    config.meta["test_keys_file"] = base_path / "split_keys" / "test_keys.pkl"
    
    # Regularization settings
    if args.edge_dropout:
        config.backbone.edge_dropout = args.edge_dropout
    if args.dropout:
        config.backbone.dropout = args.dropout

    # Set data config
    config.batch_size = args.batch_size

    # Set up dataset
    config.train_dataset = DC.adsorption_db_config(base_path, "train", args=args)
    config.val_dataset = DC.adsorption_db_config(base_path, "val", args=args)
    config.test_dataset = DC.adsorption_db_config(base_path, "test", args=args)

    graph_classification_loss_coefficients: dict[AdsorptionDbTarget, float] = {}

    # config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_balanced_accuracy", mode="max")
    config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_mae", mode="min")

    config.graph_scalar_targets = []
    config.graph_classification_adabin_targets = []    

    for idx, target in enumerate(targets):

        # Set up loss coefficients
        try:
            graph_classification_loss_coefficients[target] = args.targets_loss_coefficients[idx]
        except:
            graph_classification_loss_coefficients[target] = config.graph_classification_loss_coefficient_default
            
    
    
        config.graph_classification_adabin_targets.append(
            MulticlassClassificationTargetConfig(
                name=target, 
                num_classes=args.adabin_num_classes[idx],
                # class_weights=args.adabin_class_weights,
                class_weights=args.adabin_num_classes[idx] * [1.0],
                dropout=args.adabin_dropout,
            )
        )
    
    
    config.graph_classification_loss_coefficients = cast(
        dict[str, float], graph_classification_loss_coefficients)
    
    
def equiformer_v2_adsorption_db_config_adabin_(
        config: AdsorptionDbConfig,
        targets: list[AdsorptionDbTarget],
        base_path: Path,
        args: argparse.Namespace = argparse.Namespace(),
        # lr_base: float = 8.0e-5
    ):
    # Optimizer settings
    # config.optimizer = AdamWConfig(
    #     lr=args.lr,
    #     amsgrad=False,
    #     betas=(0.9, 0.95),
    #     weight_decay=args.weight_decay,
    # )
    
    # Assign key file(s) path
    config.meta["train_keys_file"] = base_path / "split_keys" / "train_keys.pkl"
    config.meta["val_keys_file"] = base_path / "split_keys" / "val_keys.pkl"
    config.meta["test_keys_file"] = base_path / "split_keys" / "test_keys.pkl"
    
    # # Regularization settings
    # if args.edge_dropout:
    #     config.backbone.edge_dropout = args.edge_dropout
    # if args.dropout:
    #     config.backbone.dropout = args.dropout

    # Set data config
    config.batch_size = args.batch_size
    
    # Set max_num_elements 
    config.backbone.max_num_elements = 100
    # if args.checkpoint_tag == "odac_public" or "autoreg" in args.checkpoint_tag:
    #     config.backbone.max_num_elements = 100
    
    # Set up dataset
    config.train_dataset = DC.adsorption_db_config(base_path, "train", args=args)
    config.val_dataset = DC.adsorption_db_config(base_path, "val", args=args)
    config.test_dataset = DC.adsorption_db_config(base_path, "test", args=args)

    graph_classification_loss_coefficients: dict[AdsorptionDbTarget, float] = {}

    # config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_balanced_accuracy", mode="max")
    config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_mae", mode="min")

    config.graph_scalar_targets = []
    config.graph_classification_adabin_targets = []    

    for idx, target in enumerate(targets):

        # Set up loss coefficients
        try:
            graph_classification_loss_coefficients[target] = args.targets_loss_coefficients[idx]
        except:
            graph_classification_loss_coefficients[target] = config.graph_classification_loss_coefficient_default
            
    
    
        config.graph_classification_adabin_targets.append(
            MulticlassClassificationTargetConfig(
                name=target, 
                num_classes=args.adabin_num_classes[idx],
                # class_weights=args.adabin_class_weights,
                class_weights=args.adabin_num_classes[idx] * [1.0],
                dropout=args.adabin_dropout,
            )
        )
    
    
    config.graph_classification_loss_coefficients = cast(
        dict[str, float], graph_classification_loss_coefficients)
    
def autoreg_llama_equiformer_v2_adsorption_db_config_adabin_(
        config: AdsorptionDbConfig,
        targets: list[AdsorptionDbTarget],
        base_path: Path,
        args: argparse.Namespace = argparse.Namespace(),
        # lr_base: float = 8.0e-5
    ):
    # Optimizer settings
    # config.optimizer = AdamWConfig(
    #     lr=args.lr,
    #     amsgrad=False,
    #     betas=(0.9, 0.95),
    #     weight_decay=args.weight_decay,
    # )
    
    # Assign key file(s) path
    config.meta["train_keys_file"] = base_path / "split_keys" / "train_keys.pkl"
    config.meta["val_keys_file"] = base_path / "split_keys" / "val_keys.pkl"
    config.meta["test_keys_file"] = base_path / "split_keys" / "test_keys.pkl"
    
    # # Regularization settings
    # if args.edge_dropout:
    #     config.backbone.edge_dropout = args.edge_dropout
    # if args.dropout:
    #     config.backbone.dropout = args.dropout

    # Set data config
    config.batch_size = args.batch_size
    
    # Set max_num_elements 
    config.backbone.max_num_elements = 100
    # if args.checkpoint_tag == "odac_public" or "autoreg" in args.checkpoint_tag:
    #     config.backbone.max_num_elements = 100
    
    # Set up dataset
    config.train_dataset = DC.adsorption_db_config(base_path, "train", args=args)
    config.val_dataset = DC.adsorption_db_config(base_path, "val", args=args)
    config.test_dataset = DC.adsorption_db_config(base_path, "test", args=args)

    graph_classification_loss_coefficients: dict[AdsorptionDbTarget, float] = {}

    # config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_balanced_accuracy", mode="max")
    config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_mae", mode="min")

    config.graph_scalar_targets = []
    config.graph_classification_adabin_targets = []    

    for idx, target in enumerate(targets):

        # Set up loss coefficients
        try:
            graph_classification_loss_coefficients[target] = args.targets_loss_coefficients[idx]
        except:
            graph_classification_loss_coefficients[target] = config.graph_classification_loss_coefficient_default
            
    
    
        config.graph_classification_adabin_targets.append(
            MulticlassClassificationTargetConfig(
                name=target, 
                num_classes=args.adabin_num_classes[idx],
                # class_weights=args.adabin_class_weights,
                class_weights=args.adabin_num_classes[idx] * [1.0],
                dropout=args.adabin_dropout,
            )
        )
    
    
    config.graph_classification_loss_coefficients = cast(
        dict[str, float], graph_classification_loss_coefficients)
    

def jmp_l_adsorption_db_multi_task_config_(
        config: AdsorptionDbConfig,
        targets: list[AdsorptionDbTarget],
        args: argparse.Namespace = argparse.Namespace(),
    ):
    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=args.lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    config.metrics.report_mape = False
    config.metrics.report_smape = True
    config.metrics.report_roi_smape = True
    config.metrics.report_rmse = False
    config.metrics.report_mse = False
    
    # Assign key file(s) path
    # config.meta["train_keys_file"] = base_path / "split_keys" / "train_keys.pkl"
    # config.meta["val_keys_file"] = base_path / "split_keys" / "val_keys.pkl"
    # config.meta["test_keys_file"] = base_path / "split_keys" / "test_keys.pkl"
    
    # Regularization settings
    if args.edge_dropout:
        config.backbone.edge_dropout = args.edge_dropout
    if args.dropout:
        config.backbone.dropout = args.dropout

    # Set data config
    config.batch_size = args.batch_size

    # Set up dataset
    # config.train_dataset = DC.adsorption_db_config(base_path, "train", args=args)
    # config.val_dataset = DC.adsorption_db_config(base_path, "val", args=args)
    # config.test_dataset = DC.adsorption_db_config(base_path, "test", args=args)
    
    paths = load_env_paths()
    root_path = paths["root"]

    c2261_path = root_path / "datasets"
    # Handle train_samples_limit: <= 0 means no limit, 0-1 means percentage, > 1 means absolute count
    if (train_samples_limit := args.train_samples_limit) <= 0:
        train_samples_limit = None  # None means the full dataset will be used
    if (val_samples_limit := args.val_samples_limit) < 0:
        val_samples_limit = None
    if (test_samples_limit := args.test_samples_limit) < 0:
        test_samples_limit = None

    dataset_names = args.tasks

    
    
    # TODO: Allow for differnt target loss scaling per task
    # currently, every target (in every task) has the same loss coefficient
    graph_scalar_loss_coefficients: dict[AdsorptionDbTarget, float] = {}
    for idx, target in enumerate(targets):
        try:
            graph_scalar_loss_coefficients[target] = args.targets_loss_coefficients[idx]
        except:
            graph_scalar_loss_coefficients[target] = config.graph_scalar_loss_coefficient_default

    # Make sure we only optimize for the selected targets
    config.graph_scalar_targets = cast(list[str], targets)
    config.graph_scalar_loss_coefficients = cast(
        dict[str, float], graph_scalar_loss_coefficients)
    _apply_graph_scalar_reduction_to_config(config, targets, args)
    
    all_tasks = {
        "adsorption_db1_merged": TaskConfig(
            name="adsorption_db_1",
            train_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/mof_db_1/lmdb/train",
                sample_n=DatasetSampleNConfig(
                    sample_n=train_samples_limit,
                    seed=args.seed
                ) if train_samples_limit is not None else None,
                args=args,
            ) if "adsorption_db1_merged" in dataset_names else None,
            val_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/mof_db_1/lmdb/val",
                sample_n=DatasetSampleNConfig(
                    sample_n=val_samples_limit,
                    seed=args.seed
                ) if val_samples_limit is not None else None,
                args=args,
            ),
            test_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/mof_db_1/lmdb/test",
                sample_n=DatasetSampleNConfig(
                    sample_n=test_samples_limit,
                    seed=args.seed
                ) if test_samples_limit is not None else None,
                args=args,
            ),
            graph_scalar_targets=config.graph_scalar_targets,
            graph_scalar_loss_coefficients=config.graph_scalar_loss_coefficients,
            normalization=_build_normalization_map("adsorption_db1_merged", targets, args),
        ),

        "adsorption_db2_merged": TaskConfig(
            name="adsorption_db_2",
            train_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/mof_db_2/lmdb/train",
                sample_n=DatasetSampleNConfig(
                    sample_n=train_samples_limit,
                    seed=args.seed
                ) if train_samples_limit is not None else None,
                args=args,
            ) if "adsorption_db2_merged" in dataset_names else None,
            val_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/mof_db_2/lmdb/val",
                sample_n=DatasetSampleNConfig(
                    sample_n=val_samples_limit,
                    seed=args.seed
                ) if val_samples_limit is not None else None,
                args=args,
            ),
            test_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/mof_db_2/lmdb/test",
                sample_n=DatasetSampleNConfig(
                    sample_n=test_samples_limit,
                    seed=args.seed
                ) if test_samples_limit is not None else None,
                args=args,
            ),
            graph_scalar_targets=config.graph_scalar_targets,
            graph_scalar_loss_coefficients=config.graph_scalar_loss_coefficients,
            normalization=_build_normalization_map("adsorption_db2_merged", targets, args),
        ),


        "adsorption_cof_db2": TaskConfig(
            name="adsorption_cof_db2",
            train_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/cof_db_2/lmdb/train",
                sample_n=DatasetSampleNConfig(
                    sample_n=train_samples_limit,
                    seed=args.seed
                ) if train_samples_limit is not None else None,
                args=args,
            ) if "adsorption_cof_db2" in dataset_names else None,
            val_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/cof_db_2/lmdb/val",
                sample_n=DatasetSampleNConfig(
                    sample_n=val_samples_limit,
                    seed=args.seed
                ) if val_samples_limit is not None else None,
                args=args,
            ),
            test_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/cof_db_2/lmdb/test",
                sample_n=DatasetSampleNConfig(
                    sample_n=test_samples_limit,
                    seed=args.seed
                ) if test_samples_limit is not None else None,
                args=args,
            ),
            graph_scalar_targets=config.graph_scalar_targets,
            graph_scalar_loss_coefficients=config.graph_scalar_loss_coefficients,
            normalization=_build_normalization_map("adsorption_cof_db2", targets, args),
        )

    }

    config.train_tasks = [task for task in all_tasks.values() if task.train_dataset is not None]
    config.tasks = [task for task in all_tasks.values()]
    config.trainer.num_sanity_val_steps = 0
    # config.primary_metric = PrimaryMetricConfig(name=f"{config.tasks[0].name}/{targets[0]}_mae", mode="min")
    config.primary_metric = PrimaryMetricConfig(name="loss", mode="min")
    # config.trainer.use_distributed_sampler = False

    # Handle taskifying graph-level keys
    config.mt_dataset = MTDatasetConfig(
        taskify_keys_graph=config.graph_scalar_targets,
        taskify_keys_node=[],
        sample_type=args.sample_type,
        sample_temperature=args.sample_temperature,
        strict=True,
    )

    config.use_balanced_batch_sampler = False
    config.pin_memory = True
    config.num_workers = getattr(args, "num_workers", 2)
    # config.persistent_workers = True
    # config.prefetch_factor = 4
    # config.task_loss_scaling = True



def equiformer_v2_adsorption_db_multitask_config_(
        config: AdsorptionDbConfig,
        targets: list[AdsorptionDbTarget],
        args: argparse.Namespace = argparse.Namespace(),
    ):
    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=args.lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Feature fusion settings
    config.mof_feature_fusion = MOFFeatureConfig(
        enabled=args.enable_feature_fusion,
        feature_name=args.fusion_feature_name,
        fusion_type=args.fusion_type,
        hidden_dim=args.fusion_hidden_dim,
    )

    
    # Assign key file(s) path
    # config.meta["train_keys_file"] = base_path / "split_keys" / "train_keys.pkl"
    # config.meta["val_keys_file"] = base_path / "split_keys" / "val_keys.pkl"
    # config.meta["test_keys_file"] = base_path / "split_keys" / "test_keys.pkl"
    
    # Set data config
    config.batch_size = args.batch_size

    # Set up dataset
    # config.train_dataset = DC.adsorption_db_config(base_path, "train", args=args)
    # config.val_dataset = DC.adsorption_db_config(base_path, "val", args=args)
    # config.test_dataset = DC.adsorption_db_config(base_path, "test", args=args)
    config.metrics.report_mape = False
    config.metrics.report_smape = True
    config.metrics.report_roi_smape = True
    config.metrics.report_rmse = False
    config.metrics.report_mse = False
    config.trainer.num_sanity_val_steps = 0
    # config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_mae", mode="min") 

    paths = load_env_paths()
    root_path = paths["root"]
    
    c2261_path = root_path / "datasets"
    # Handle train_samples_limit: <= 0 means no limit, 0-1 means percentage, > 1 means absolute count
    if (train_samples_limit := args.train_samples_limit) <= 0:
        train_samples_limit = None  # None means the full dataset will be used
    if (val_samples_limit := args.val_samples_limit) < 0:
        val_samples_limit = None
    if (test_samples_limit := args.test_samples_limit) < 0:
        test_samples_limit = None

    if len(args.tasks) == 1 and " " in args.tasks[0]:
        dataset_names = args.tasks[0].split()
    else:
        dataset_names = args.tasks

    
    
    # TODO: Allow for differnt target loss scaling per task
    # currently, every target (in every task) has the same loss coefficient
    # this has been done imo.
    graph_scalar_loss_coefficients: dict[AdsorptionDbTarget, float] = {}
    for idx, target in enumerate(targets):
        try:
            graph_scalar_loss_coefficients[target] = args.targets_loss_coefficients[idx]
        except:
            graph_scalar_loss_coefficients[target] = config.graph_scalar_loss_coefficient_default

    # Make sure we only optimize for the selected targets
    config.graph_scalar_targets = cast(list[str], targets)
    config.graph_scalar_loss_coefficients = cast(
        dict[str, float], graph_scalar_loss_coefficients)
    _apply_graph_scalar_reduction_to_config(config, targets, args)
    


    all_tasks = {}
    
    all_tasks["adsorption_db1_merged"] = TaskConfig(
        name="adsorption_db_1",
        train_dataset=FinetuneLmdbDatasetConfig(
            src= c2261_path / "all-adsorption/mof_db_1/lmdb/train",
            sample_n=DatasetSampleNConfig(
                sample_n=train_samples_limit,
                seed=args.seed
            ) if train_samples_limit is not None else None,
            args=args,
        ) if "adsorption_db1_merged" in dataset_names else None,
        val_dataset=FinetuneLmdbDatasetConfig(
            src= c2261_path / "all-adsorption/mof_db_1/lmdb/val",
            sample_n=DatasetSampleNConfig(
                sample_n=val_samples_limit,
                seed=args.seed
            ) if val_samples_limit is not None else None,
            args=args,
        ),
        test_dataset=FinetuneLmdbDatasetConfig(
            src= c2261_path / "all-adsorption/mof_db_1/lmdb/test",
            sample_n=DatasetSampleNConfig(
                sample_n=test_samples_limit,
                seed=args.seed
            ) if test_samples_limit is not None else None,
            args=args,
        ),
        graph_scalar_targets=config.graph_scalar_targets,
        graph_scalar_loss_coefficients=config.graph_scalar_loss_coefficients,
        normalization=_build_normalization_map("adsorption_db1_merged", targets, args),
    )

    # Only include DB2 if directory exists
    db2_path = c2261_path / "all-adsorption/mof_db_2"
    if db2_path.exists():
        all_tasks["adsorption_db2_merged"] = TaskConfig(
            name="adsorption_db_2",
            train_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/mof_db_2/lmdb/train",
                sample_n=DatasetSampleNConfig(
                    sample_n=train_samples_limit,
                    seed=args.seed
                ) if train_samples_limit is not None else None,
                args=args,
            ) if "adsorption_db2_merged" in dataset_names else None,
            val_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/mof_db_2/lmdb/val",
                sample_n=DatasetSampleNConfig(
                    sample_n=val_samples_limit,
                    seed=args.seed
                ) if val_samples_limit is not None else None,
                args=args,
            ),
            test_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/mof_db_2/lmdb/test",
                sample_n=DatasetSampleNConfig(
                    sample_n=test_samples_limit,
                    seed=args.seed
                ) if test_samples_limit is not None else None,
                args=args,
            ),
            graph_scalar_targets=config.graph_scalar_targets,
            graph_scalar_loss_coefficients=config.graph_scalar_loss_coefficients,
            normalization=_build_normalization_map("adsorption_db2_merged", targets, args),
        )

    cof2_path = c2261_path / "all-adsorption/cof_db_2"
    if cof2_path.exists():
        all_tasks["adsorption_cof_db2"] = TaskConfig(
            name="adsorption_cof_db2",
            train_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/cof_db_2/lmdb/train",
                sample_n=DatasetSampleNConfig(
                    sample_n=train_samples_limit,
                    seed=args.seed
                ) if train_samples_limit is not None else None,
                args=args,
            ) if "adsorption_cof_db2" in dataset_names else None,
            val_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/cof_db_2/lmdb/val",
                sample_n=DatasetSampleNConfig(
                    sample_n=val_samples_limit,
                    seed=args.seed
                ) if val_samples_limit is not None else None,
                args=args,
            ),
            test_dataset=FinetuneLmdbDatasetConfig(
                src= c2261_path / "all-adsorption/cof_db_2/lmdb/test",
                sample_n=DatasetSampleNConfig(
                    sample_n=test_samples_limit,
                    seed=args.seed
                ) if test_samples_limit is not None else None,
                args=args,
            ),
            graph_scalar_targets=config.graph_scalar_targets,
            graph_scalar_loss_coefficients=config.graph_scalar_loss_coefficients,
            normalization=_build_normalization_map("adsorption_cof_db2", targets, args),
        )

    # If val_same_as_train is set, remove any task not in dataset_names
    if args.val_same_as_train:
        allowed = set(dataset_names)

        # Remove any task whose key is not in allowed
        removed = []
        for key in list(all_tasks.keys()):
            if key not in allowed:
                removed.append(key)
                del all_tasks[key]
        
        print(f"Keeping only datasets used for training. kept={list(all_tasks.keys())}, removed={removed}")


    # for name, task in all_tasks.items():
    #     if task.train_dataset is None:
    #         raise ValueError(
    #             f"Training dataset for '{name}' is missing. Verify --tasks argument. "
    #             f"Current --tasks is: {args.tasks}"
    #         )

    config.train_tasks = [task for task in all_tasks.values() if task.train_dataset is not None]
    config.tasks = [task for task in all_tasks.values()]
    # config.primary_metric = PrimaryMetricConfig(name=f"{config.tasks[0].name}/{targets[0]}_mae", mode="min")
    config.primary_metric = PrimaryMetricConfig(name="loss", mode="min")
    # config.trainer.use_distributed_sampler = False

    config.heteroscedastic = getattr(args, "heteroscedastic", False)
    config.hetero_nll_weight = getattr(args, "hetero_nll_weight", 1.0)
    config.hetero_std_weight = getattr(args, "hetero_std_weight", 0.1)
    config.hetero_min_var = getattr(args, "hetero_min_var", 1e-6)

    if config.heteroscedastic:
        config.std_targets = [f"{t}_std" for t in targets]

    # Binary classification for high-value detection
    config.enable_classification = getattr(args, "enable_classification", False)
    config.classification_threshold = getattr(args, "classification_threshold", 0.5)
    config.classification_targets = getattr(args, "classification_targets", [])
    config.classification_loss_weight = getattr(args, "classification_loss_weight", 1.0)

    # Focal loss parameters for class imbalance
    config.focal_gamma = getattr(args, "focal_gamma", 2.0)
    config.focal_alpha_pos = getattr(args, "focal_alpha_pos", 0.75)
    config.focal_alpha_neg = getattr(args, "focal_alpha_neg", 0.25)

    # Asymmetric regression loss for discovery (penalize under-prediction)
    config.asymmetric_regression = getattr(args, "asymmetric_regression", False)
    config.asymmetric_tau = getattr(args, "asymmetric_tau", 0.9)

    config.mt_dataset = MTDatasetConfig(
        taskify_keys_graph=config.graph_scalar_targets,
        taskify_keys_node=[],
        sample_type=args.sample_type,
        sample_temperature=args.sample_temperature,
        strict=True,
    )

    config.use_balanced_batch_sampler = False
    config.pin_memory = True
    # config.trainer.precision = "16-mixed"
    # config.trainer.set_float32_matmul_precision = "medium"
    # config.trainer.check_val_every_n_epoch = 2
    # config.task_loss_scaling = True
