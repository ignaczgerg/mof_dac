#!/usr/bin/env python3
"""
Merge shard outputs written as .npz files with keys: sid (object array) and pred (numeric array).

Usage:
    python merge_shards.py /path/to/base_out_dir

Behavior:
    - Finds shard subfolders in BASE matching: shard_<id> or shard_<id>_of_<N>
    - For each *.npz filename that exists in any shard, concatenates the arrays
      in ascending shard_id order and writes to BASE/merged/<filename>.
    - Writes BASE/merged/merge_summary.json and a MERGE_OK sentinel.
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


_SHARD_RX = re.compile(r"^shard_(\d+)(?:_of_(\d+))?$")


def _find_shard_dirs(base: Path) -> List[Path]:
    dirs = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        m = _SHARD_RX.match(p.name)
        if m:
            shard_id = int(m.group(1))
            total = int(m.group(2)) if m.group(2) is not None else None
            dirs.append((shard_id, total, p))
    if not dirs:
        dirs = [(-1, None, p) for p in base.iterdir() if p.is_dir() and p.name.startswith("shard_")]
    dirs.sort(key=lambda t: (t[0], t[2].name))
    return [p for _, __, p in dirs]


def _group_files_by_name(shard_dirs: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for sd in shard_dirs:
        for f in sd.glob("*.npz"):
            groups.setdefault(f.name, []).append(f)
    for name, paths in groups.items():
        paths.sort(key=lambda p: p.parent.name)
    return groups


def _concat_check_dim(arrs: List[np.ndarray], key: str) -> np.ndarray:
    if not arrs:
        raise ValueError(f"No arrays for key {key}")
    ref = arrs[0].shape[1:]
    for a in arrs[1:]:
        if a.shape[1:] != ref:
            raise ValueError(
                f"Shape mismatch for '{key}': {arrs[0].shape} vs {a.shape} (beyond axis 0)"
            )
    return np.concatenate(arrs, axis=0)


def merge_base(base_dir: Path, merged_subdir: str = "merged") -> Path:
    shard_dirs = _find_shard_dirs(base_dir)
    if not shard_dirs:
        raise SystemExit(f"No shard directories found under: {base_dir}")

    groups = _group_files_by_name(shard_dirs)
    if not groups:
        raise SystemExit(f"No .npz files found under shard directories in: {base_dir}")

    out_dir = base_dir / merged_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "base": str(base_dir),
        "merged_dir": str(out_dir),
        "num_shards_found": len(shard_dirs),
        "files_merged": {},
    }

    for fname, parts in groups.items():
        sid_parts, pred_parts = [], []
        total_rows = 0
        for part in parts:
            d = np.load(part, allow_pickle=True)
            sid = d["sid"]
            pred = d["pred"]
            sid_parts.append(sid)
            pred_parts.append(pred)
            total_rows += sid.shape[0]

        sid_all = _concat_check_dim(sid_parts, "sid")
        pred_all = _concat_check_dim(pred_parts, "pred")

        out_path = out_dir / fname
        np.savez_compressed(out_path, sid=sid_all, pred=pred_all)
        summary["files_merged"][fname] = {
            "parts": len(parts),
            "rows": int(total_rows),
            "out": str(out_path),
        }

    (out_dir / "MERGE_OK").write_text("done\n", encoding="utf-8")
    (out_dir / "merge_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Merge shard .npz outputs into a single folder.")
    ap.add_argument("base", type=Path, help="Base directory containing shard_* subfolders")
    ap.add_argument("--merged-subdir", type=str, default="merged",
                    help="Name of the output subdirectory (default: merged)")
    args = ap.parse_args()

    out = merge_base(args.base, args.merged_subdir)
    print(f"[OK] merged outputs written to: {out}")


if __name__ == "__main__":
    main()
