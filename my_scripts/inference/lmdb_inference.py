from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import json
import pandas as pd
import re
import os
import torch
from jmp.tasks.finetune.adsorption_db import AdsorptionDbModel, AdsorptionDbConfig
from jmp.tasks.finetune import EquiformerV2ModelWrapper
from jmp.models.equiformer_v2.config import EquiformerV2Config
from jmp.utils.finetune_state_dict import load_equiformer_ema_weights
from utils import CIFFolderDataset
from tqdm import tqdm

from utils import (
    preselect_cifs_below,
    run_predict,
    compute_geometric_properties_batch,
)



def main():
    p = argparse.ArgumentParser("On-the-fly CIF inference (no LMDB)")
    p.add_argument("--ckpt", required=True, help="Path to fine-tuned .ckpt")
    p.add_argument("--cif_dir", required=True, type=Path, help="Directory containing CIF files")
    p.add_argument("--pattern", type=str, default=r".*\.cif$", help="Regex to match CIF files in --cif_dir")
    p.add_argument("--list_file", type=Path, default=None,
                   help="Optional: .txt/.list or .csv listing CIFs (names or stems).")
    p.add_argument("--list_col", type=str, default=None, help="If list_file is CSV, column with CIF names/stems")
    p.add_argument("--sid_from", choices=["stem", "name"], default="stem", help="How to construct SID")
    p.add_argument("--fail_on_error", action="store_true", help="Raise on CIF read errors instead of skipping")

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory. Default: <ckpt_dir>/predictions_external/cif_infer")
    p.add_argument("--max_atoms", type=int, default=1000,
               help="Only process CIFs with <= this many atoms. Set <0 to disable.")
    p.add_argument("--prefilter_workers", type=int, default=8,
                help="Threads for prefiltering CIFs by atom count.")
    p.add_argument("--limit_files",type=int,default=-1,
                help="Only consider the first N matched CIFs (after sorting). Use -1 for no limit."
            )
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling/filtering")
    p.add_argument("--shard_size", type=int, default=None,
                   help="Number of files per shard (contiguous block). If set, overrides --limit_files.")
    p.add_argument("--exclude_csv", type=Path, default=None,
               help="CSV whose first column lists CIF names/stems to EXCLUDE from processing.")

    args = p.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Infer num_distance_basis from the checkpoint's state_dict tensor shapes
    state_dict = raw["state_dict"]
    offset_key = "backbone.distance_expansion.offset"
    if offset_key in state_dict:
        inferred_num_distance_basis = state_dict[offset_key].shape[0]
        print(f"[INFO] Inferred num_distance_basis from checkpoint: {inferred_num_distance_basis}")
    else:
        inferred_num_distance_basis = None
        print("[WARN] Could not infer num_distance_basis from checkpoint")

    hp = raw.get("hyper_parameters", {})

    if isinstance(hp, dict):
        backbone_obj = hp.get("backbone")
        if isinstance(backbone_obj, dict):
            if inferred_num_distance_basis is not None:
                backbone_obj["num_distance_basis"] = inferred_num_distance_basis
            backbone_name = backbone_obj.get("name", "equiformer_v2")
            if backbone_name in ("EquiformerV2", "equiformer_v2"):
                hp["backbone"] = EquiformerV2Config(**backbone_obj)
        elif hasattr(backbone_obj, "num_distance_basis"):
            if inferred_num_distance_basis is not None and backbone_obj.num_distance_basis != inferred_num_distance_basis:
                print(f"[INFO] Overriding backbone.num_distance_basis: {backbone_obj.num_distance_basis} -> {inferred_num_distance_basis}")
                backbone_obj.num_distance_basis = inferred_num_distance_basis
        cfg = AdsorptionDbConfig(**hp)
    elif isinstance(hp, AdsorptionDbConfig):
        cfg = hp
        if inferred_num_distance_basis is not None and cfg.backbone.num_distance_basis != inferred_num_distance_basis:
            print(f"[INFO] Overriding backbone.num_distance_basis: {cfg.backbone.num_distance_basis} -> {inferred_num_distance_basis}")
            cfg.backbone.num_distance_basis = inferred_num_distance_basis
    else:
        raise RuntimeError(f"Unexpected hyper_parameters type: {type(hp)}")

    print(f"[INFO] Final config backbone.num_distance_basis: {cfg.backbone.num_distance_basis}")
    print(f"[INFO] use_charge_embedding: {getattr(cfg.backbone, 'use_charge_embedding', False)}")
    model = AdsorptionDbModel(cfg)
    model.load_state_dict(raw["state_dict"], strict=True)
    if cfg.model_cls == EquiformerV2ModelWrapper and cfg.meta.get("ema_backbone", False):
        load_equiformer_ema_weights(raw, model)
    model.config.args.log_predictions = True
    model.config.trainer.num_sanity_val_steps = 0
    model.config.trainer.logging.wandb.enabled = False

    if not hasattr(model.config.args, "max_neighbors") or model.config.args.max_neighbors is None:
        model.config.args.max_neighbors = getattr(cfg.backbone, "max_neighbors", 32) # hardcoded
    if not hasattr(model.config.args, "cutoff") or model.config.args.cutoff is None:
        model.config.args.cutoff = getattr(cfg.backbone, "max_radius", 6.0) # hardcoded
    if not hasattr(model.config.args, "no_pbc"):
        model.config.args.no_pbc = not getattr(cfg.backbone, "use_pbc", True)

    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )

    from jmp.datasets.finetune.base import LmdbDataset
    d = Path('/opt/resources/datasets/adsorption_aramco/lmdb/test')
    args.enable_feature_fusion = True
    args.fusion_feature_names = ["pld", "lcd", "gcd", "unitcell_volume", "density", "asa", "av", "nav"]
    
    ds = LmdbDataset(
        src=str(d),
        metadata_path=str(d / "metadata.npz"),
        args=args,
        split_name="train",
    )

    out_dir = args.out

    run_predict(
        model=model,
        dataset=ds,
        device=device,
        out_dir=out_dir,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
if __name__ == "__main__":
    main()
