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
from ase.io.cif import parse_cif
from jmp.tasks.finetune.adsorption_db import AdsorptionDbModel, AdsorptionDbConfig, TaskConfig


# ─────────────────────────────────────────────────────────────────────
# MOF geometric property computation (pyzeo + ASE)
# ─────────────────────────────────────────────────────────────────────

# Avogadro's number
_N_A = 6.02214076e23
# 1 Å³ = 1e-24 cm³
_ANG3_TO_CM3 = 1e-24


def compute_mof_geometric_properties(
    cif_path: str | Path,
    probe_radius: float = 1.86,
    mc_samples: int = 5000,
) -> dict[str, float] | None:
    """Compute MOF textural/geometric properties from a CIF file.

    Uses pyzeo for LCD, PLD, GCD, ASA, AV, NAV and ASE for
    unitcell_volume and density.

    Returns dict with keys: lcd, pld, gcd, unitcell_volume, density,
    asa, av, nav.  Returns None on failure.
    """
    import tempfile
    import os
    from pyzeo.netstorage import AtomNetwork
    from pyzeo.area_volume import surface_area as pyzeo_surface_area
    from pyzeo.area_volume import volume as pyzeo_volume

    cif_path = str(cif_path)

    try:
        atoms = ase_read(cif_path, index=0, format="cif")
        volume_ang3 = float(atoms.cell.volume)
        mass_amu = float(atoms.get_masses().sum())
        mass_g = mass_amu / _N_A
        volume_cm3 = volume_ang3 * _ANG3_TO_CM3
        density = mass_g / volume_cm3 if volume_cm3 > 0 else 0.0
    except Exception as e:
        print(f"[WARN] compute_mof_geometric_properties: ASE failed for {cif_path}: {e}",
              file=sys.stderr, flush=True)
        return None

    try:
        atmnet = AtomNetwork.read_from_CIF(cif_path, rad_flag=True)
    except Exception as e:
        print(f"[WARN] compute_mof_geometric_properties: pyzeo CIF load failed for {cif_path}: {e}",
              file=sys.stderr, flush=True)
        return None

    # --- LCD, PLD, GCD via free sphere parameters ---
    lcd, pld, gcd = 0.0, 0.0, 0.0
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".res")
        os.close(fd)
        atmnet.calculate_free_sphere_parameters(tmp_path)
        with open(tmp_path) as f:
            line = f.read().strip()
        os.unlink(tmp_path)
        # Format: "filename    Di Df Dif"
        parts = line.split()
        if len(parts) >= 4:
            lcd = float(parts[-3])   # Di  (largest included sphere)
            pld = float(parts[-2])   # Df  (largest free sphere)
            gcd = float(parts[-1])   # Dif (included sphere along free sphere path)
    except Exception as e:
        print(f"[WARN] compute_mof_geometric_properties: free sphere params failed for {cif_path}: {e}",
              file=sys.stderr, flush=True)

    # --- ASA (accessible surface area, m²/g) ---
    asa = 0.0
    try:
        sa_str = pyzeo_surface_area(atmnet, probe_radius, probe_radius, mc_samples)
        if isinstance(sa_str, bytes):
            sa_str = sa_str.decode("utf-8")
        asa_m2_cm3 = _parse_pyzeo_field(sa_str, "ASA_m^2/cm^3:")
        asa = asa_m2_cm3 / density if density > 0 else 0.0
    except Exception as e:
        print(f"[WARN] compute_mof_geometric_properties: surface_area failed for {cif_path}: {e}",
              file=sys.stderr, flush=True)

    # --- AV, NAV (accessible / non-accessible volume, cm³/g) ---
    av, nav = 0.0, 0.0
    try:
        vol_str = pyzeo_volume(atmnet, probe_radius, probe_radius, mc_samples)
        if isinstance(vol_str, bytes):
            vol_str = vol_str.decode("utf-8")
        av_frac = _parse_pyzeo_field(vol_str, "AV_Volume_fraction:")
        nav_frac = _parse_pyzeo_field(vol_str, "NAV_Volume_fraction:")
        # cm³/g = volume_fraction * volume_cm³ / mass_g
        if density > 0:
            av = av_frac / density
            nav = nav_frac / density
    except Exception as e:
        print(f"[WARN] compute_mof_geometric_properties: volume failed for {cif_path}: {e}",
              file=sys.stderr, flush=True)

    return {
        "lcd": lcd,
        "pld": pld,
        "gcd": gcd,
        "unitcell_volume": volume_ang3,
        "density": density,
        "asa": asa,
        "av": av,
        "nav": nav,
    }


def _parse_pyzeo_field(output_str: str, field_name: str) -> float:
    """Parse a named numeric field from pyzeo's output string."""
    for token_idx, token in enumerate(output_str.split()):
        if token == field_name:
            tokens = output_str.split()
            if token_idx + 1 < len(tokens):
                return float(tokens[token_idx + 1])
    return 0.0


def compute_geometric_properties_batch(
    cif_paths: Sequence[Path],
    probe_radius: float = 1.86,
    mc_samples: int = 5000,
    n_workers: int = 4,
    timeout: int = 120,
) -> dict[str, dict[str, float]]:
    """Compute geometric properties for a batch of CIF files.

    Returns dict mapping CIF stem -> property dict.
    Uses multiprocessing with maxtasksperchild=1 for crash isolation.
    """
    import multiprocessing as mp

    results: dict[str, dict[str, float]] = {}
    tasks = [(str(p), probe_radius, mc_samples) for p in cif_paths]

    n_workers = min(n_workers, len(cif_paths))
    if n_workers <= 1:
        for p in tqdm(cif_paths, desc="Geometric properties"):
            props = compute_mof_geometric_properties(str(p), probe_radius, mc_samples)
            if props is not None:
                results[p.stem] = props
    else:
        with mp.Pool(processes=n_workers, maxtasksperchild=1) as pool:
            async_results = [
                (p, pool.apply_async(_geometric_worker, (t,)))
                for p, t in zip(cif_paths, tasks)
            ]
            for p, ar in tqdm(async_results, desc="Geometric properties"):
                try:
                    props = ar.get(timeout=timeout)
                    if props is not None:
                        results[p.stem] = props
                except mp.TimeoutError:
                    print(f"[WARN] Geometric properties timeout for {p.stem}")
                except Exception as e:
                    print(f"[WARN] Geometric properties failed for {p.stem}: {e}")

    return results


def _geometric_worker(args):
    """Picklable worker for multiprocessing geometric property computation."""
    cif_path, probe_radius, mc_samples = args
    return compute_mof_geometric_properties(cif_path, probe_radius, mc_samples)


def _read_partial_charges(cif_path: str | Path):
    """Extract per-atom partial charges from a CIF file, if present.

    Returns a numpy float32 array of shape (n_atoms,) or None if the CIF
    does not contain ``_atom_type_partial_charge`` in the atom_site loop.
    """
    with open(cif_path, "r") as f:
        for block in parse_cif(f):
            raw = block.get("_atom_type_partial_charge")
            if raw is not None:
                return np.array([float(v) for v in raw], dtype=np.float32)
    return None


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
    if norm_cfg is None:
        return value
    denorm_method = getattr(norm_cfg, "denormalize", None)
    if callable(denorm_method):
        return denorm_method(value)
    return value

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
                if target not in preds:
                    continue

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
        geometric_properties: dict[str, dict[str, float]] | None = None,
        fusion_feature_names: list[str] | None = None,
    ):
        self.cif_dir = Path(cif_dir)
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.sid_from = sid_from
        self.fail_on_error = bool(fail_on_error)
        self.geometric_properties = geometric_properties or {}
        self.fusion_feature_names = fusion_feature_names

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

        # Set precomputed geometric properties (lcd, pld, gcd, etc.)
        # and assemble into mof_descriptor tensor for feature fusion
        geo = self.geometric_properties.get(path.stem)
        if geo is not None:
            for key, value in geo.items():
                setattr(d, key, float(value))
            if self.fusion_feature_names is not None:
                values = [float(geo.get(fn, 0.0)) for fn in self.fusion_feature_names]
                d.mof_descriptor = torch.tensor([values], dtype=torch.float32)  # [1, F]

        # Read per-atom partial charges from CIF (_atom_type_partial_charge)
        charges = _read_partial_charges(str(path))
        if charges is not None:
            d.partial_charges = torch.tensor(charges, dtype=torch.float32)

        return d