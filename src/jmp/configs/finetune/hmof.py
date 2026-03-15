from __future__ import annotations
from pathlib import Path
from typing import cast, Iterable
import argparse

from jmp.tasks.finetune.hmof import HMofConfig, HMofTarget
from jmp.modules.transforms.normalize import NormalizationConfig, PositionNormalizationConfig
from jmp.tasks.finetune import dataset_config as DC 

try:
    from jmp.tasks.finetune.base import PrimaryMetricConfig, RLPConfig, WarmupCosRLPConfig
except Exception:
    PrimaryMetricConfig = None  
    RLPConfig = None            
    WarmupCosRLPConfig = None   


def _as_list(x, n: int, *, coerce=float, default=None):
    if x is None:
        return None if default is None else [default] * n
    if isinstance(x, str):
        tokens = [t for t in x.replace(",", " ").split() if t]
        if len(tokens) == 1:
            try:
                return [coerce(tokens[0])] * n
            except Exception:
                return [tokens[0]] * n
        vals = []
        for t in tokens:
            try:
                vals.append(coerce(t))
            except Exception:
                vals.append(t)
        if len(vals) == 1:
            return vals * n
        if len(vals) == n:
            return vals
        raise ValueError(f"Argument '{x}' does not align with n={n}")
    if not isinstance(x, Iterable) or isinstance(x, (int, float)):
        try:
            return [coerce(x)] * n
        except Exception:
            return [x] * n
    x = list(x)
    if len(x) == 1:
        try:
            return [coerce(x[0])] * n
        except Exception:
            return [x[0]] * n
    if len(x) == n:
        try:
            return [coerce(v) for v in x]
        except Exception:
            return x
    raise ValueError(f"Provided iterable length {len(x)} != n={n}")


def equiformer_v2_hmof_config_(
    config: HMofConfig,
    base_path: Path,
    targets: list[HMofTarget],
    args: argparse.Namespace = argparse.Namespace(),
):
    config.batch_size = getattr(args, "batch_size", 8)
    config.backbone.use_pbc = bool(getattr(args, "use_pbc", True))
    config.backbone.max_radius = float(getattr(args, "cutoff", 8.0))
    config.backbone.max_neighbors = int(getattr(args, "max_neighbors", 5)) 
    config.backbone.max_num_elements = int(getattr(args, "max_num_elements", 100))

    if hasattr(config.backbone, "avg_num_nodes"):
        config.backbone.avg_num_nodes = float(getattr(args, "avg_num_nodes", 60))
    if hasattr(config.backbone, "avg_degree"):
        config.backbone.avg_degree = float(getattr(args, "avg_degree", 16))

    config.train_dataset = DC.hmof_config(base_path, "train", args=args)
    config.val_dataset   = DC.hmof_config(base_path, "val",   args=args)
    config.test_dataset  = DC.hmof_config(base_path, "test",  args=args)

    config.graph_scalar_targets = cast(list[str], targets)

    n = len(targets)
    norm_types = _as_list(getattr(args, "normalization_type", None), n, coerce=str, default="standard")
    means      = _as_list(getattr(args, "norm_mean", None),            n, coerce=float)
    stds       = _as_list(getattr(args, "norm_std", None),             n, coerce=float)

    config.normalization = {}
    if norm_types is not None or means is not None or stds is not None:
        if (means is None) ^ (stds is None):
            raise ValueError("Provide both --norm-mean and --norm-std or neither.")
        if means is None and stds is None:
            means = [0.0] * n
            stds  = [1.0] * n
        for i, t in enumerate(targets):
            config.normalization[t] = NormalizationConfig(
                type="default",
                mean=float(means[i]),
                std=float(stds[i]),
                normalization_type=str(norm_types[i]),
            )

    if bool(getattr(args, "position_norm", False)):
        pos_std = float(getattr(args, "pos_std", 2.5))
        config.normalization["pos"] = PositionNormalizationConfig(
            mean=0.0, std=pos_std, normalization_type="standard"
        )

    coefs = _as_list(getattr(args, "targets_loss_coefficients", None), n, coerce=float)
    if coefs is not None:
        config.graph_scalar_loss_coefficients = cast(
            dict[str, float], {t: float(c) for t, c in zip(targets, coefs)}
        )

    if PrimaryMetricConfig is not None:
        config.primary_metric = PrimaryMetricConfig(name=f"{targets[0]}_mae", mode="min")

    if WarmupCosRLPConfig is not None and hasattr(args, "epochs"):
        config.lr_scheduler = WarmupCosRLPConfig(
            warmup_epochs=int(getattr(args, "warmup_epochs", 5)),
            warmup_start_lr_factor=float(getattr(args, "warmup_start_lr_factor", 0.002)),
            should_restart=bool(getattr(args, "should_restart", False)),
            max_epochs=int(args.epochs),
            min_lr_factor=float(getattr(args, "min_lr_factor", 0.002)),
            rlp=(RLPConfig(patience=int(getattr(args, "rlp_patience", 10)),
                           factor=float(getattr(args, "rlp_factor", 0.1)))
                 if RLPConfig is not None else None),
        )

    if not hasattr(config, "meta") or config.meta is None:
        config.meta = {}
    config.meta.setdefault("pbc_default", config.backbone.use_pbc)
