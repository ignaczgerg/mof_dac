from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

from jmp.lightning import Trainer
from jmp.tasks.finetune.adsorption_db import AdsorptionDbConfig, AdsorptionDbModel
from jmp.utils.env_utils import detect_env

DEFAULT_LOG_ROOT = Path("/ibex/project/c2261/dac_iclr/finetune/lightning_logs")
DEFAULT_OUT_ROOT = Path("/ibex/project/c2261/dac_iclr/finetune/ablation_eval")


def _namespacefy_args(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        obj = {k: _namespacefy_args(v) for k, v in obj.items()}
        if "args" in obj and isinstance(obj["args"], Mapping):
            obj["args"] = Namespace(**obj["args"])
        return obj
    if isinstance(obj, list):
        return [_namespacefy_args(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_namespacefy_args(v) for v in obj)
    return obj


def _extract_config_dict(ckpt: dict) -> dict:
    hp = ckpt.get("hyper_parameters", {})
    if isinstance(hp, Mapping) and "hparams" in hp and isinstance(hp["hparams"], Mapping):
        cfg_dict = hp["hparams"]
    elif isinstance(hp, Mapping) and "config" in hp and isinstance(hp["config"], Mapping):
        cfg_dict = hp["config"]
    elif isinstance(hp, Mapping):
        cfg_dict = hp
    else:
        raise RuntimeError("No usable hyper-parameters found in checkpoint.")
    return dict(cfg_dict)


def build_replace_map(paths_yaml_path: Path, current_env: str) -> dict[str, str]:
    envs = yaml.safe_load(open(paths_yaml_path))["environments"]
    other_env = "docker" if current_env == "ibex" else "ibex"
    return {
        str(envs[other_env]["root"]): str(envs[current_env]["root"]),
        str(envs[other_env]["ocp"]): str(envs[current_env]["ocp"]),
        str(envs[other_env]["pt_logging"]): str(envs[current_env]["pt_logging"]),
        str(envs[other_env]["ft_logging"]): str(envs[current_env]["ft_logging"]),
    }


def replace_paths(obj: Any, replace_map: dict[str, str]) -> Any:
    if isinstance(obj, (str, Path)):
        s = str(obj)
        for old, new in replace_map.items():
            if s.startswith(old):
                return type(obj)(s.replace(old, new, 1))
        return obj
    if isinstance(obj, Namespace):
        for k, v in obj.__dict__.items():
            setattr(obj, k, replace_paths(v, replace_map))
        return obj
    if isinstance(obj, dict):
        return {k: replace_paths(v, replace_map) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_paths(v, replace_map) for v in obj]
    if isinstance(obj, tuple):
        return tuple(replace_paths(v, replace_map) for v in obj)
    return obj


def find_checkpoint(run_dir: Path, run_tag: str) -> Path:
    preferred = [
        run_dir / "aramco_ablation" / run_tag / "checkpoints" / "last.ckpt",
        run_dir / "checkpoints" / "last.ckpt",
        run_dir / "checkpoints" / "step_latest.ckpt",
    ]
    for p in preferred:
        if p.exists():
            return p
    candidates = sorted(run_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No checkpoint found for {run_tag} under {run_dir}")


def split_outputs_exist(pred_root: Path, split: str) -> bool:
    return any((pred_root / "final" / split).glob("*/*.npz"))


def main() -> None:
    ap = argparse.ArgumentParser(
        "Re-run inference from an ablation checkpoint and dump per-structure predictions"
    )
    ap.add_argument("--run_tag")
    ap.add_argument("--ckpt")
    ap.add_argument("--log_root", default=str(DEFAULT_LOG_ROOT))
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["val", "test"])
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--precision", default="32")
    ap.add_argument("--max_natoms", type=int, default=None)
    ap.add_argument("--limit_batches", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.run_tag and not args.ckpt:
        ap.error("pass --run_tag (and/or --ckpt)")

    log_root = Path(args.log_root)
    run_tag = args.run_tag
    if args.ckpt:
        ckpt_path = Path(args.ckpt).resolve()
        if not run_tag:
            run_tag = ckpt_path.parent.parent.name
    else:
        run_dir = log_root / run_tag
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        ckpt_path = find_checkpoint(run_dir, run_tag)

    out_run_dir = Path(args.out_root) / run_tag
    pred_root = out_run_dir / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    todo = [s for s in args.splits if args.overwrite or not split_outputs_exist(pred_root, s)]
    if not todo:
        print(f"[skip] {run_tag}: predictions already present for {args.splits} in {pred_root}")
        return

    print(f"[info] run_tag={run_tag}")
    print(f"[info] checkpoint={ckpt_path}")
    print(f"[info] out={pred_root} splits={todo}")

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    raw = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = _namespacefy_args(_extract_config_dict(raw))
    cfg_dict = replace_paths(cfg_dict, build_replace_map(REPO_ROOT / "paths.yaml", detect_env()))
    cfg_dict["trainer"]["precision"] = args.precision

    cfg = AdsorptionDbConfig(**cfg_dict)
    model = AdsorptionDbModel(cfg)
    missing, unexpected = model.load_state_dict(raw["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}")

    bin_edges = raw.get("bin_edges")
    if bin_edges is not None:
        try:
            model.bin_edges = [t for t in bin_edges]
        except Exception as exc:
            print(f"[warn] could not restore bin_edges: {exc}")

    model.config.args.log_predictions = True
    model.config.args.num_workers = int(args.num_workers)
    if args.max_natoms is not None:
        model.config.args.max_natoms = args.max_natoms
    try:
        model.config.num_workers = int(args.num_workers)
    except Exception:
        pass
    try:
        model.config.trainer.num_sanity_val_steps = 0
        model.config.trainer.logging.wandb.enabled = False
        model.config.trainer.logging.csv.enabled = False
        model.config.trainer.logging.tensorboard.enabled = False
        model.config.trainer.logging.enabled = False
        model.config.trainer.logger = False
    except Exception:
        pass
    try:
        model.config.trainer.auto_set_default_root_dir = False
        model.config.trainer.default_root_dir = str(out_run_dir)
    except Exception:
        pass

    model._get_results_dir = lambda trainer, _d=str(pred_root): _d

    trainer_kwargs = {}
    if args.limit_batches is not None:
        limit = int(args.limit_batches) if args.limit_batches > 1 else args.limit_batches
        trainer_kwargs["limit_val_batches"] = limit
        trainer_kwargs["limit_test_batches"] = limit

    trainer = Trainer(
        config=model.config,
        use_distributed_sampler=False,
        accelerator="gpu" if use_cuda else "cpu",
        devices=1,
        num_nodes=1,
        callbacks=[],
        enable_checkpointing=False,
        **trainer_kwargs,
    )

    if "val" in todo:
        trainer.validate(model)
    if "test" in todo:
        trainer.test(model)

    written = sorted(pred_root.glob("final/*/*/*.npz"))
    if not written:
        print(f"[error] no prediction files written under {pred_root}", file=sys.stderr)
        sys.exit(1)
    for p in written:
        print(f"[ok] {p}")


if __name__ == "__main__":
    main()
