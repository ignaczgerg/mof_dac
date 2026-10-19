from __future__ import annotations

import argparse
import json
import os
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from ase.io import read as ase_read
from torch.utils.data import Dataset, SequentialSampler, Subset
from torch_geometric.data import Data
from tqdm import tqdm

import jmp
from jmp.modules.atom_batch_sampler import AtomBucketBatchSampler
from jmp.modules.dataset import dataset_transform as DT
from jmp.tasks.finetune.adsorption_db import (
    UPTAKE_TARGETS,
    AdsorptionDbConfig,
    AdsorptionDbModel,
    _convert_uptake_mmolg_to_per_cell,
)

BUCKETS = (128, 256, 384, 512, 768, 1024, 1536, 2048)


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
    if isinstance(hp, Mapping) and isinstance(hp.get("hparams"), Mapping):
        return dict(hp["hparams"])
    if isinstance(hp, Mapping) and isinstance(hp.get("config"), Mapping):
        return dict(hp["config"])
    return dict(hp)


def load_model(ckpt: Path, device: torch.device) -> AdsorptionDbModel:
    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg_dict = _namespacefy_args(_extract_config_dict(raw))
    cfg_dict["trainer"]["precision"] = "32"
    model = AdsorptionDbModel(AdsorptionDbConfig(**cfg_dict))
    model.load_state_dict(raw["state_dict"], strict=True)
    if model.config.heteroscedastic:
        raise RuntimeError("heteroscedastic checkpoints are not supported by this script")
    if model.config.uptake_conversion_mode not in ("none", "per_cell"):
        raise RuntimeError(f"uptake_conversion_mode={model.config.uptake_conversion_mode} not supported")
    if len(model.config.tasks) != 1:
        raise RuntimeError("expected a single-task checkpoint")
    return model.eval().to(device)


def _count_atoms(path: str) -> tuple[str, int | None, str | None]:
    try:
        return path, int(len(ase_read(path, index=0, format="cif"))), None
    except Exception as exc:
        return path, None, f"{type(exc).__name__}: {exc}"


class CIFShardDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Data:
        path = self.paths[i]
        atoms = ase_read(str(path), index=0, format="cif")
        zs = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        d = Data(
            pos=torch.tensor(atoms.get_positions(), dtype=torch.float32),
            cell=torch.tensor(atoms.cell.array, dtype=torch.float32),
            atomic_numbers=zs,
            natoms=int(len(zs)),
            pbc=torch.tensor(atoms.pbc, dtype=torch.bool),
            idx=int(i),
            id=path.stem,
            sid=path.stem,
        )
        return _convert_uptake_mmolg_to_per_cell(d)


def predict(model: AdsorptionDbModel, dataset, sizes: list[int], cap: int, num_workers: int, device: torch.device):
    cfg = model.config
    task = cfg.tasks[0]
    norms = task.normalization or {}
    batch_sampler = AtomBucketBatchSampler(
        base_sampler=SequentialSampler(range(len(sizes))),
        sizes=sizes,
        max_atoms_per_batch=cap,
        bucket_boundaries=BUCKETS,
        drop_last=False,
        seed=0,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=model.collate_fn,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    out = {t: {"sid": [], "pred": []} for t in cfg.graph_scalar_targets}
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", unit="batch"):
            batch = batch.to(device)
            preds = model(batch)
            B = batch.num_graphs
            sids = list(batch.sid) if isinstance(batch.sid, (list, tuple)) else [batch.sid]
            if len(sids) != B:
                raise RuntimeError(f"sid count {len(sids)} != num_graphs {B}")
            for target in cfg.graph_scalar_targets:
                y = preds[target].reshape(B, -1)[:, 0].double().cpu().numpy()
                norm = norms.get(target)
                if norm is not None:
                    mean = float(norm["mean"] if isinstance(norm, Mapping) else norm.mean)
                    std = float(norm["std"] if isinstance(norm, Mapping) else norm.std)
                    ntype = (norm.get("normalization_type") if isinstance(norm, Mapping)
                             else getattr(norm, "normalization_type", "standard"))
                    y = y * std + mean
                    if ntype == "log":
                        y = np.exp(y)
                if cfg.uptake_conversion_mode == "per_cell" and target in UPTAKE_TARGETS:
                    y = y * 1000.0 / batch.cell_mol_weight.reshape(-1).double().cpu().numpy()
                out[target]["sid"].extend(str(s) for s in sids)
                out[target]["pred"].extend(y.tolist())
            n_batches += 1
    return out, n_batches


def validate(args, device):
    model = load_model(args.ckpt, device)
    ds = model.test_dataset()
    base = model._test_dataset_base
    n = min(args.validate_lmdb, len(ds))
    idx = list(range(n))
    sizes = [int(base[i].natoms) for i in idx]
    out, _ = predict(model, Subset(ds, idx), sizes, args.max_atoms_per_batch, args.num_workers, device)
    report = {}
    for target, d in out.items():
        ref = np.load(Path(args.reference_dir) / f"{target}.npz", allow_pickle=True)
        ref_map = dict(zip(ref["sid"].astype(str).tolist(), ref["pred"].astype(np.float64).tolist()))
        pairs = np.array([(p, ref_map[s]) for s, p in zip(d["sid"], d["pred"]) if s in ref_map])
        diff = np.abs(pairs[:, 0] - pairs[:, 1])
        rel = diff / (np.abs(pairs[:, 1]) + 1e-12)
        report[target] = {"matched": int(len(pairs)), "of": len(d["sid"]), "max_abs": float(diff.max()),
                          "median_rel": float(np.median(rel)), "p99_rel": float(np.percentile(rel, 99))}
    print("[VALIDATE] " + json.dumps(report))


def main() -> None:
    p = argparse.ArgumentParser("CIF campaign shard with an adsorption_aramco ablation checkpoint")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--shard_id", type=int, default=None)
    p.add_argument("--shard_size", type=int)
    p.add_argument("--max_natoms", type=int, required=True)
    p.add_argument("--max_atoms_per_batch", type=int, required=True)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefilter_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--validate_lmdb", type=int, default=None)
    p.add_argument("--reference_dir", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda")
    if args.validate_lmdb:
        validate(args, device)
        return

    t0 = time.time()
    shard_id = args.shard_id if args.shard_id is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    names = [ln.strip() for ln in args.manifest.read_text().splitlines() if ln.strip()]
    cif_dir = Path(names[0])
    files = names[1:]
    total = len(files)
    num_shards = (total + args.shard_size - 1) // args.shard_size
    start = shard_id * args.shard_size
    if start >= total:
        print(f"[INFO] shard {shard_id} empty (start={start} >= total={total})")
        return
    shard_files = files[start:start + args.shard_size]
    if args.limit is not None:
        shard_files = shard_files[:args.limit]

    out_dir = args.out / f"shard_{shard_id:04d}_of_{num_shards:04d}"
    if (out_dir / "DONE").exists():
        print(f"[SKIP] {out_dir} already has DONE")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] jmp={jmp.__file__}")
    print(f"[INFO] shard {shard_id}/{num_shards} files {start}:{start + len(shard_files)} of {total}")

    kept: list[Path] = []
    sizes: list[int] = []
    dropped: dict[str, str] = {}
    paths = [str(cif_dir / f) for f in shard_files]
    with ProcessPoolExecutor(max_workers=args.prefilter_workers) as ex:
        for path, n, err in tqdm(ex.map(_count_atoms, paths, chunksize=16), total=len(paths), desc="Prefilter CIFs", unit="file"):
            stem = Path(path).stem
            if n is None:
                dropped[stem] = f"unreadable: {err}"
            elif n == 0:
                dropped[stem] = "natoms=0"
            elif n >= args.max_natoms:
                dropped[stem] = f"natoms={n}>={args.max_natoms}"
            else:
                kept.append(Path(path))
                sizes.append(n)
    print(f"[INFO] prefilter kept {len(kept)} dropped {len(dropped)} in {time.time() - t0:.1f}s")

    model = load_model(args.ckpt, device)
    ds = DT.transform(CIFShardDataset(kept), model.test_data_transform)
    out, n_batches = predict(model, ds, sizes, args.max_atoms_per_batch, args.num_workers, device)

    expected = {pth.stem for pth in kept}
    for target, d in out.items():
        if len(d["sid"]) != len(expected) or set(d["sid"]) != expected:
            raise RuntimeError(f"{target}: predicted {len(d['sid'])} sids, expected {len(expected)}")
        np.savez_compressed(out_dir / f"{target}.npz",
                            sid=np.array(d["sid"], dtype=object), pred=np.array(d["pred"], dtype=np.float32))

    meta = {
        "shard_id": shard_id, "num_shards": num_shards, "shard_size": args.shard_size, "total_files": total,
        "files_in_shard": len(shard_files), "kept": len(kept), "dropped": len(dropped),
        "max_natoms_strict_lt": args.max_natoms, "max_atoms_per_batch": args.max_atoms_per_batch,
        "batches": n_batches, "ckpt": str(args.ckpt), "jmp": jmp.__file__,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "dropped.json").write_text(json.dumps(dropped, indent=1))
    (out_dir / "shard_meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "DONE").write_text("ok\n")
    print(f"[DONE] {json.dumps(meta)}")


if __name__ == "__main__":
    main()
