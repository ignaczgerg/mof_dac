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

from utils import (
    preselect_cifs_below,
    run_predict,
)


# Matches labels like O215, C1024, Si122
_LABEL_TOKEN = re.compile(r"\b([A-Z][a-z]?)(\d{2,})(?:[A-Za-z]*)\b")

def _load_exclude_stems_from_csv(csv_path: Path) -> set[str]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    col0 = df.iloc[:, 0]
    stems: set[str] = set()
    for v in col0:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        stems.add(Path(s).stem.lower())
    return stems 

def _env_int(name: str) -> int | None:
    v = os.getenv(name)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None

def _discover_cif_paths(
    cif_dir: Path,
    pattern: str,
    list_file: Path | None,
    list_col: str | None,
    exclude_csv: Path | None = None,
) -> List[Path]:
    exclude_stems: set[str] | None = None
    if exclude_csv is not None:
        exclude_stems = _load_exclude_stems_from_csv(Path(exclude_csv))

    if list_file is None:
        rx = re.compile(pattern, re.IGNORECASE)
        paths = sorted([p for p in Path(cif_dir).iterdir() if p.is_file() and rx.match(p.name)])
        if exclude_stems:
            paths = [p for p in paths if p.stem.lower() not in exclude_stems]
        return paths

    lf = Path(list_file)
    if not lf.is_file():
        raise FileNotFoundError(f"List file not found: {lf}")

    if lf.suffix.lower() in {".txt", ".list"}:
        names = [ln.strip() for ln in lf.read_text(encoding="utf-8").splitlines() if ln.strip()]
        paths: List[Path] = []
        for name in names:
            p = Path(name)
            if not p.suffix:
                p = Path(cif_dir) / f"{name}.cif"
            if not p.is_file():
                raise FileNotFoundError(f"Missing CIF path referenced in list: {p}")
            paths.append(p)
        if exclude_stems:
            paths = [p for p in paths if p.stem.lower() not in exclude_stems]
        return paths

    df = pd.read_csv(lf)
    if list_col is None:
        cand = [c for c in df.columns if "cif" in c.lower() or "id" in c.lower()]
        if not cand:
            raise ValueError("CSV list provided but --list_col not set and no obvious column found.")
        list_col = cand[0]
    vals = df[list_col].astype(str).tolist()
    paths: List[Path] = []
    for v in vals:
        p = Path(cif_dir) / v
        if p.suffix == "":
            p = p.with_suffix(".cif")
        if not p.is_file():
            raise FileNotFoundError(f"Missing CIF path referenced in CSV: {p}")
        paths.append(p)
    if exclude_stems:
        paths = [p for p in paths if p.stem.lower() not in exclude_stems]
    return paths


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
    p.add_argument("--shard_id", type=int, default=None,
                   help="This task's shard index (0-based). Defaults to SLURM_ARRAY_TASK_ID if present, else 0.")
    p.add_argument("--num_shards", type=int, default=None,
                   help="Total shard count. Optional (used for metadata/validation).")
    p.add_argument("--shard_size", type=int, default=None,
                   help="Number of files per shard (contiguous block). If set, overrides --limit_files.")
    p.add_argument("--exclude_csv", type=Path, default=None,
               help="CSV whose first column lists CIF names/stems to EXCLUDE from processing.")

    args = p.parse_args()

    candidate_paths = _discover_cif_paths(
        cif_dir=args.cif_dir,
        pattern=args.pattern,
        list_file=args.list_file,
        list_col=args.list_col,
        exclude_csv=args.exclude_csv
    )

    total_files = len(candidate_paths)

    shard_id = args.shard_id
    if shard_id is None:
        shard_id = _env_int("SLURM_ARRAY_TASK_ID")
    if shard_id is None:
        shard_id = 0

    shard_size = args.shard_size
    num_shards = args.num_shards
    if shard_size is not None and shard_size > 0 and num_shards is None:
        num_shards = (total_files + shard_size - 1) // shard_size

    if shard_size is not None and shard_size > 0:
        start = shard_id * shard_size
        end = min(start + shard_size, total_files)
        if start >= total_files:
            print(f"[INFO] Shard {shard_id}/{num_shards or '?'} empty (start={start} >= total={total_files}). Exiting.")
            return
        candidate_paths = candidate_paths[start:end]
        effective_limit = None
    else:
        effective_limit = args.limit_files if (args.limit_files is not None and args.limit_files > 0) else None
        if effective_limit is not None:
            candidate_paths = candidate_paths[:effective_limit]

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

    if args.max_atoms is not None and args.max_atoms >= 0:
        filtered_paths = preselect_cifs_below(candidate_paths, args.max_atoms, workers=args.prefilter_workers)
    else:
        filtered_paths = candidate_paths

    if shard_size is not None and shard_size > 0:
        print(f"[INFO] CIFs considered in shard {shard_id}/{num_shards or '?'}: "
              f"{len(filtered_paths)} (slice size target={shard_size}, total={total_files})")
    else:
        print(f"[INFO] CIFs considered: {len(filtered_paths)} (limit={effective_limit})")

    ds = CIFFolderDataset(
        cif_dir=args.cif_dir,
        pattern=args.pattern,
        list_file=None,
        list_col=None,
        sid_from=args.sid_from,
        fail_on_error=args.fail_on_error,
        paths=filtered_paths,
    )

    base_out_dir = args.out or (ckpt_path.parent / "predictions_external" / "cif_infer")
    if shard_size is not None and shard_size > 0:
        if num_shards is None:
            shard_dir = base_out_dir / f"shard_{shard_id:04d}"
        else:
            shard_dir = base_out_dir / f"shard_{shard_id:04d}_of_{int(num_shards):04d}"
        out_dir = shard_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "shard_meta.json", "w") as f:
            json.dump({
                "total_files": total_files,
                "shard_id": shard_id,
                "num_shards": num_shards,
                "shard_size": shard_size,
            }, f, indent=2)
    else:
        out_dir = base_out_dir

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
