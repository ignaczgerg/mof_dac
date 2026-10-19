# preprocess.py
import os
import argparse
import pickle
import logging
import warnings
from typing import Dict, List, Tuple

import pandas as pd
from ase.io import read as ase_read

import torch
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

LOGGER = logging.getLogger("db2-1-preprocess")


def _apply_warning_filters():
    warnings.filterwarnings(
        "ignore",
        message=r".*crystal system .* is not interpreted for space group 1.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"scaled_positions \d+ and \d+ are equivalent",
    )


def _read_cif_worker(task):
    """
    Worker for ProcessPoolExecutor.
    Args: task -> (idx, cif_path)
    Returns: (idx, pos, cell, zs, pbc) or (idx, exc)
    """
    _apply_warning_filters()
    idx, cif_path = task
    try:
        atoms = ase_read(cif_path, index=0, format="cif")
        pos = atoms.get_positions()
        cell = atoms.cell.array
        zs = atoms.get_atomic_numbers()
        pbc = atoms.pbc
        return (idx, pos, cell, zs, pbc)
    except Exception as e:
        return (idx, e)


class DataLoader(object):
    def __init__(
        self,
        base_folder,
        split_keys,
        entity_key,
        num_workers: int = 0,
        rebuild_lut: bool = False,   # kept for CLI compatibility (unused)
        log_every: int = 5,
        backend: str = "thread",
    ):
        """
        Minimal loader:
          - CSV IDs are used verbatim.
          - CIF path is <base>/cifs/<ID>.cif (case-insensitive on filename).
        """
        self.entity_key = entity_key
        self.cif_path = os.path.join(base_folder, "cifs")
        self.csv_path = os.path.join(base_folder, "db2_merged.csv")
        self.df = pd.read_csv(self.csv_path)
        self.log_every = max(1, int(log_every))
        self.backend = backend if backend in ("thread", "process") else "thread"

        # Optional properties passthrough (kept as-is; remove if unneeded)
        self.properties = {
            "CO2_Qst_kJmol-1": "qst_co2",
            "CO2_KH_molkgPa": "kh_co2",
            "CO2_uptake_mmolg-1": "co2_uptake",
            "H2O_Qst_kJmol-1": "qst_h2o",
            "H2O_KH_molkgPa": "kh_h2o",
            "Selectivity_CO2_H2O": "selectivity_co2_h2o",
        }

        # Index all .cif files once (case-insensitive), keyed by stem (without .cif)
        if not os.path.isdir(self.cif_path):
            raise FileNotFoundError(f"CIF folder not found: {self.cif_path}")
        file_index: Dict[str, str] = {}
        for fn in os.listdir(self.cif_path):
            if fn.lower().endswith(".cif"):
                stem = fn[:-4]  # remove '.cif'
                file_index[stem.lower()] = fn

        # Build ID -> filename map (strict: require a matching file)
        unique_ids: List[str] = self.df[self.entity_key].astype(str).tolist()
        self._id2file: Dict[str, str] = {}
        missing: List[str] = []
        for raw in set(unique_ids):
            key = raw.strip()
            hit = file_index.get(key.lower())
            if hit is None:
                missing.append(raw)
            else:
                self._id2file[key] = hit

        if missing:
            examples = ", ".join(missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} CSV IDs have no matching '<ID>.cif' in {self.cif_path}. "
                f"Examples: {examples}"
            )

        if split_keys is None:
            self.split_keys = list(range(len(self.df)))
        else:
            self.split_keys = list(split_keys)

        self.num_workers = max(0, int(num_workers))

    def __len__(self):
        return len(self.split_keys)

    def _lookup_cif_path(self, entity_id_raw: str) -> str:
        try:
            fn = self._id2file[entity_id_raw]
        except KeyError:
            # Fallback to case-insensitive key (shouldn't happen if built above)
            fn = self._id2file.get(entity_id_raw.lower())  # type: ignore[arg-type]
            if fn is None:
                raise KeyError(f"No CIF for ID '{entity_id_raw}'")
        return os.path.join(self.cif_path, fn)

    def _build_one(self, pos_idx: int) -> Tuple[int, Data]:
        row = self.df.iloc[pos_idx]
        entity_id = str(row[self.entity_key])
        cif_path = self._lookup_cif_path(entity_id)

        atoms = ase_read(cif_path, index=0, format="cif")
        data_object = Data(
            pos=torch.tensor(atoms.get_positions(), dtype=torch.float32),
            cell=torch.tensor(atoms.cell.array, dtype=torch.float32),
            atomic_numbers=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
            natoms=int(len(atoms)),
            pbc=torch.tensor(atoms.pbc, dtype=torch.bool),
            idx=pos_idx,
            id=entity_id,
            sid=entity_id,
            **{short: row[csv_col] for csv_col, short in self.properties.items() if csv_col in row},
        )
        return pos_idx, data_object

    def __iter__(self):
        total = len(self.split_keys)
        LOGGER.info(
            "Starting iteration over %d items (workers=%d, backend=%s)...",
            total, self.num_workers, self.backend,
        )

        if self.backend == "thread":
            if self.num_workers >= 2:
                with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                    for k, (pos_idx, data_object) in enumerate(ex.map(self._build_one, self.split_keys), 1):
                        if k % self.log_every == 0 or k == total:
                            LOGGER.info(" processed %d/%d (%.1f%%)", k, total, 100.0 * k / total)
                        yield pos_idx, data_object
            else:
                for k, pos_idx in enumerate(self.split_keys, 1):
                    out = self._build_one(pos_idx)
                    if k % self.log_every == 0 or k == total:
                        LOGGER.info(" processed %d/%d (%.1f%%)", k, total, 100.0 * k / total)
                    yield out
            return

        if self.backend == "process" and self.num_workers >= 2:
            def _tasks():
                for idx in self.split_keys:
                    entity_id = str(self.df.iloc[idx][self.entity_key])
                    cif_path = self._lookup_cif_path(entity_id)
                    yield (idx, cif_path)

            with ProcessPoolExecutor(max_workers=self.num_workers, initializer=_apply_warning_filters) as ex:
                for k, result in enumerate(ex.map(_read_cif_worker, _tasks()), 1):
                    idx, payload = result[0], result[1:]
                    if len(payload) == 1 and isinstance(payload[0], Exception):
                        LOGGER.warning("Skipping idx=%s due to CIF read error: %s", idx, payload[0])
                        continue
                    pos, cell, zs, pbc = payload
                    row = self.df.iloc[idx]
                    entity_id = str(row[self.entity_key])
                    data_object = Data(
                        pos=torch.tensor(pos, dtype=torch.float32),
                        cell=torch.tensor(cell, dtype=torch.float32),
                        atomic_numbers=torch.tensor(zs, dtype=torch.long),
                        natoms=int(len(zs)),
                        pbc=torch.tensor(pbc, dtype=torch.bool),
                        idx=idx,
                        id=entity_id,
                        sid=entity_id,
                        **{short: row[csv_col] for csv_col, short in self.properties.items() if csv_col in row},
                    )
                    if k % self.log_every == 0 or k == total:
                        LOGGER.info(" processed %d/%d (%.1f%%)", k, total, 100.0 * k / total)
                    yield idx, data_object
            return

        for k, idx in enumerate(self.split_keys, 1):
            out = self._build_one(idx)
            if k % self.log_every == 0 or k == total:
                LOGGER.info(" processed %d/%d (%.1f%%)", k, total, 100.0 * k / total)
            yield out


def _configure_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("ase").setLevel(logging.WARNING)
    _apply_warning_filters()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--entity_key", required=True)
    parser.add_argument("--split_keys", type=int, nargs="+", default=None)
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--backend", type=str, default="thread", choices=["thread", "process"])
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--rebuild_lut", action="store_true")  # accepted but unused

    args = parser.parse_args()
    _configure_logging(args.log_level)

    LOGGER.info("Initializing DataLoader...")
    dl = DataLoader(
        base_folder=args.dataset_path,
        entity_key=args.entity_key,
        split_keys=args.split_keys,
        num_workers=args.num_workers,
        log_every=args.log_every,
        backend=args.backend,
        rebuild_lut=args.rebuild_lut,
    )

    count = 0
    for _, _ in dl:
        count += 1
    LOGGER.info("Finished preprocessing %d items.", count)