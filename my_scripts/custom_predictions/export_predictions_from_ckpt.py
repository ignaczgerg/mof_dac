import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import os
from argparse import Namespace
from typing import Mapping, Any
import shutil
from jmp.lightning import Trainer
from jmp.tasks.finetune.adsorption_db import AdsorptionDbModel, AdsorptionDbConfig
from torch.utils.data import DataLoader as TorchDataLoader, SequentialSampler
from jmp.utils.env_utils import detect_env
import yaml


if "SLURM_JOB_ID" in os.environ:
    os.environ["SUBMITIT_EXECUTOR"] = "slurm"

def _eval_clone_no_drop(dl):
    """
    Rebuild a DataLoader for evaluation:
      - sequential iteration over the same dataset
      - keep the batch_size & collate_fn from the original
      - ensure drop_last=False
      - no training sampler/bucket logic
    """
    return TorchDataLoader(
        dataset=dl.dataset,
        batch_size=getattr(dl, "batch_size", 1),
        sampler=SequentialSampler(dl.dataset),
        num_workers=getattr(dl, "num_workers", 0),
        collate_fn=getattr(dl, "collate_fn", None),
        pin_memory=getattr(dl, "pin_memory", False),
        persistent_workers=getattr(dl, "persistent_workers", False),
        drop_last=False,
    )


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


def build_replace_map(paths_yaml_path, current_env):
    envs = yaml.safe_load(open(paths_yaml_path))["environments"]
    other_env = "docker" if current_env == "ibex" else "ibex"

    # replace ANY occurrence of other_env base paths → current_env base paths
    return {
        str(envs[other_env]["root"]): str(envs[current_env]["root"]),
        str(envs[other_env]["ocp"]): str(envs[current_env]["ocp"]),
        str(envs[other_env]["pt_logging"]): str(envs[current_env]["pt_logging"]),
        str(envs[other_env]["ft_logging"]): str(envs[current_env]["ft_logging"]),
    }


def replace_paths(obj, replace_map):
    if isinstance(obj, (str, Path)):
        s = str(obj)
        for old, new in replace_map.items():
            if s.startswith(old):
                return type(obj)(s.replace(old, new, 1))
        return obj

    if isinstance(obj, argparse.Namespace):
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

def main():
    ap = argparse.ArgumentParser("Export predictions (val+test) from a fine-tuned checkpoint")
    ap.add_argument("--ckpt", required=True, help="Path to fine-tuned .ckpt")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--train-as-test", action="store_true",help="Run a test pass on the training set and save under final/train")
    ap.add_argument("--max_natoms", type=int, default=800, help="Filter structures with more than N atoms")

    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    ckpt_dir = ckpt_path.parent
    map_loc = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"

    raw = torch.load(str(ckpt_path), map_location="cpu")
    cfg_dict = _extract_config_dict(raw)
    cfg_dict = _namespacefy_args(cfg_dict)

    # Replace paths based on environment: ibex <-> docker
    paths_yaml = str(Path.cwd().resolve().parent.parent / "paths.yaml")
    current_env = detect_env()  # "ibex" or "docker"
    replace_map = build_replace_map(paths_yaml, current_env)
    cfg_dict = replace_paths(cfg_dict, replace_map)
    cfg_dict["trainer"]["precision"] = "32"

    # Fix backbone parameter mismatch - checkpoint was trained with num_distance_basis=600
    if "backbone" in cfg_dict:
        cfg_dict["backbone"]["num_distance_basis"] = 512

    cfg = AdsorptionDbConfig(**cfg_dict)

    model = AdsorptionDbModel(cfg)
    model.load_state_dict(raw["state_dict"], strict=True)

    if args.max_natoms is not None and hasattr(model.config, "args"):
        model.config.args.max_natoms = args.max_natoms

    # for name in ["batch_size", "eval_batch_size", "test_batch_size", "val_batch_size", "train_batch_size"]:
    #     if hasattr(model.config, name):
    #         setattr(model.config, name, 12)
    #     if hasattr(model.config, "args") and hasattr(model.config.args, name):
    #         setattr(model.config.args, name, 12)

    # if hasattr(model.config, "args"):
    #     for k, v in [("train_samples_limit", 5000),
    #                 ("val_samples_limit",   1000),
    #                 ("test_samples_limit",  1000)
    #                 ]:
    #         if hasattr(model.config.args, k):
    #             setattr(model.config.args, k, v)

    model.config.args.log_predictions = True
    try:
        model.config.trainer.num_sanity_val_steps = 0
        model.config.trainer.logging.wandb.enabled = False
    except Exception:
        pass
    if args.num_workers is not None:
        model.config.num_workers = int(args.num_workers)
        try:
            model.config.args.num_workers = int(args.num_workers)
        except Exception:
            pass

    trainer = Trainer(
        config=model.config,
        default_root_dir=str(ckpt_dir),
        use_distributed_sampler=False,
        accelerator="gpu" if map_loc == "cuda" else "cpu",
        devices=1,
        callbacks=[],
    )
    main_out_root = Path(model._get_results_dir(trainer))
    main_default_root = str(main_out_root.parent) 
    trainer.validate(model)
    if args.train_as_test:
        dl_train_raw = model.train_dataloader()

        try:
            if hasattr(dl_train_raw, "batch_sampler") and hasattr(dl_train_raw.batch_sampler, "drop_last"):
                dl_train_raw.batch_sampler.drop_last = False
        except Exception:
            pass
        
        if hasattr(model.config, "args"):
            for k in ["log_predictions_max_samples", "log_predictions_max_batches",
                    "max_logged_predictions", "predictions_max_samples"]:
                if hasattr(model.config.args, k):
                    old = getattr(model.config.args, k)
                    if old not in (None, -1):
                        setattr(model.config.args, k, -1)

        dl_train_eval = _eval_clone_no_drop(dl_train_raw)

        quick_trainer = Trainer(
            config=model.config,
            default_root_dir=main_default_root,
            use_distributed_sampler=False,
            accelerator="gpu" if map_loc == "cuda" else "cpu",
            devices=1,
            callbacks=[],
            limit_test_batches=1.0,
        )
        print(f"[info] exporting train-as-test with dataset len={len(dl_train_eval.dataset)}, "
            f"batch_size={getattr(dl_train_eval, 'batch_size', None)}")

        quick_trainer.test(model, dataloaders=dl_train_eval)

        # out_root = Path(model._get_results_dir(quick_trainer))
        split_root = (main_out_root / "final") if (main_out_root / "final").exists() else main_out_root
        test_dir  = split_root / "test"
        train_dir = split_root / "train"
        if train_dir.exists():
            print(f"[info] train predictions already at: {train_dir}")
        elif test_dir.exists():
            if train_dir.exists():
                shutil.rmtree(train_dir)
            test_dir.rename(train_dir)
            print(f"[info] moved {test_dir} -> {train_dir}")
        else:
            print(f"[warn] no predictions found at {test_dir} or {train_dir} after train-as-test.")
    
    trainer.test(model)



if __name__ == "__main__":
    main()
