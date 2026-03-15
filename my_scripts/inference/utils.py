from __future__ import annotations
import re
from pathlib import Path
import sys
from typing import Sequence, Mapping, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from ase.io import read as ase_read
from jmp.tasks.finetune.adsorption_db import AdsorptionDbModel, AdsorptionDbConfig, TaskConfig



def _namespacefy_args(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        obj = {k: _namespacefy_args(v) for k, v in obj.items()}
        return obj

    if isinstance(obj, list):
        return [_namespacefy_args(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_namespacefy_args(v) for v in obj)

    return obj

def _denorm(value: torch.Tensor, norm_cfg: Any | None) -> torch.Tensor:
    denorm_method = getattr(norm_cfg, "denormalize", None)
    if callable(denorm_method):
        return denorm_method(value)
    else:
        ValueError(f"Cannot denormalize: no valid method found in norm_cfg: {norm_cfg}")

def _head_names_and_norms(cfg: AdsorptionDbConfig) -> tuple[list[str], list[dict | None]]:
    train_tasks = getattr(cfg, "train_tasks", None)
    if train_tasks:
        names = [t.name for t in train_tasks]
        norms = [getattr(t, "normalization", None) for t in train_tasks]
        return names, norms
    else:
        ValueError(f"Cannot determine head names and norms: no train_tasks found in config.")


def _sid_to_list(sid_attr) -> List:
    if isinstance(sid_attr, (list, tuple)):
        out: List = []
        for x in sid_attr:
            out.append(x)
        return out
    else:
        UserWarning(f"[WARN] _sid_to_list: sid_attr is not list/tuple, but {type(sid_attr)}. Converting to single-item list.")
        return [sid_attr]


def _estimate_natoms(path: Path) -> int | None:
    try:
        atoms = ase_read(str(path), index=0, format="cif")
        # print(f"[INFO] _estimate_natoms: CIF {path} has {len(atoms)} atoms (ASE read)", file=sys.stderr, flush=True)
        return int(len(atoms))
    except Exception:
        print(f"[WARN] _estimate_natoms: failed to read CIF {path}", file=sys.stderr, flush=True)
        return None

def preselect_cifs_below(paths: Sequence[Path], max_atoms: int, workers: int = 1) -> List[Path]:
    keep: List[Path] = []
    for p in tqdm(paths, total=len(paths), desc="Prefilter CIFs", unit="file"):
        n = _estimate_natoms(p)
        if n is not None and n <= max_atoms:
            keep.append(p)
    return keep

def run_predict(
    model: AdsorptionDbModel,
    dataset: Dataset,
    device: torch.device,
    out_dir: Path,
    batch_size: int = 8,
    num_workers: int = 6,
):
    model.eval().to(device)
    cfg = model.config

    head_names, head_norms_base = _head_names_and_norms(cfg)
    fallback = getattr(cfg, "normalization", None) or {}
    head_norms = [norm or fallback for norm in head_norms_base]

    ds = model._apply_dataset_transforms(dataset)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
        collate_fn=model.collate_fn,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    acc = {
        head_name: {t: {"sid": [], "pred": []} for t in cfg.graph_scalar_targets}
        for head_name in head_names
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", unit="batch"):
            batch = batch.to(device)
            preds = model(batch)
            B = batch.num_graphs

            sids = _sid_to_list(getattr(batch, "sid"))
            if len(sids) != B:
                sids = sids[:B]

            for target in cfg.graph_scalar_targets:
                y = preds[target]
                y = y.view(B, -1) if (y.ndim >= 2 and y.shape[-1] > 1) else y.view(B, 1)
                C = y.shape[-1]

                for c in range(min(C, len(head_names))):
                    norm_dict = head_norms[c] or {}
                    norm_cfg = norm_dict.get(target) if isinstance(norm_dict, Mapping) else None
                    y_c = _denorm(y[:, c], norm_cfg)

                    head_name = head_names[c]
                    acc[head_name][target]["sid"].extend(sids)
                    acc[head_name][target]["pred"].extend(y_c.detach().cpu().tolist())

    out_dir.mkdir(parents=True, exist_ok=True)
    for head_name, task_acc in acc.items():
        short_name = head_name.replace("adsorption_", "")
        for target, d in task_acc.items():
            sid = np.array(d["sid"], dtype=object)
            pred = np.array(d["pred"], dtype=np.float32)
            np.savez_compressed(out_dir / f"{target}_{short_name}.npz", sid=sid, pred=pred)


class CIFFolderDataset(Dataset):
    def __init__(
        self,
        cif_dir: Path,
        pattern: str = r".*\.cif$",
        list_file: Path | None = None,
        list_col: str | None = None,
        sid_from: str = "stem",
        fail_on_error: bool = False,
        paths: Sequence[Path] | None = None,
        max_atoms: int | None = None,
        prefilter_workers: int = 8, 
    ):
        self.cif_dir = Path(cif_dir)
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.sid_from = sid_from
        self.fail_on_error = bool(fail_on_error)

        if not self.cif_dir.is_dir():
            raise FileNotFoundError(f"CIF directory not found: {self.cif_dir}")

        if paths is not None:
            candidates = [Path(p) for p in paths]
        else:
            if list_file is None:
                files = [p for p in self.cif_dir.iterdir() if p.is_file() and self.pattern.match(p.name)]
                files.sort()
                candidates = files
            else:
                list_path = Path(list_file)
                if not list_path.is_file():
                    raise FileNotFoundError(f"List file not found: {list_path}")

                if list_path.suffix.lower() in {".txt", ".list"}:
                    names: List[str] = []
                    with open(list_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            names.append(line)
                    resolved: List[Path] = []
                    for name in names:
                        p = Path(name)
                        if not p.suffix:
                            p = self.cif_dir / f"{name}.cif"
                        if not p.is_file():
                            raise FileNotFoundError(f"Missing CIF path referenced in list: {p}")
                        resolved.append(p)
                    candidates = resolved
                else:
                    df = pd.read_csv(list_path)
                    if list_col is None:
                        candidates_cols = [c for c in df.columns if "cif" in c.lower() or "id" in c.lower()]
                        if not candidates_cols:
                            raise ValueError("CSV list provided but --list_col not set and no obvious column found.")
                        list_col = candidates_cols[0]
                    vals = df[list_col].astype(str).tolist()
                    resolved: List[Path] = []
                    for v in vals:
                        p = self.cif_dir / v
                        if p.suffix == "":
                            p = p.with_suffix(".cif")
                        if not p.is_file():
                            raise FileNotFoundError(f"Missing CIF path referenced in CSV: {p}")
                        resolved.append(p)
                    candidates = resolved

        if max_atoms is not None and max_atoms >= 0:
            kept: List[Path] = []

            with ThreadPoolExecutor(max_workers=int(prefilter_workers)) as ex:
                for p, n in tqdm(ex.map(lambda q: (q, _estimate_natoms(q)), candidates),
                                 total=len(candidates), desc="Prefilter CIFs", unit="file"):
                    if n is not None and n <= max_atoms:
                        kept.append(p)
            candidates = kept

        self.items = [(i, p) for i, p in enumerate(candidates)]

    def __len__(self) -> int:
        return len(self.items)

    def _make_sid(self, path: Path) -> str:
        return path.stem if self.sid_from == "stem" else path.name

    def __getitem__(self, i: int) -> Data:
        idx, path = self.items[i]
        atoms = ase_read(str(path), index=0, format="cif")
        pos  = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
        zs   = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        pbc  = torch.tensor(atoms.pbc, dtype=torch.bool)
        sid  = self._make_sid(path)

        d = Data(
            pos=pos,
            cell=cell,
            atomic_numbers=zs,
            natoms=int(len(zs)),
            pbc=pbc,
            idx=int(idx),
            id=sid,
            sid=sid,
        )
        return d