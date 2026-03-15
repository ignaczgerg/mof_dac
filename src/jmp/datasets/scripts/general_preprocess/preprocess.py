import os
import argparse
import logging
import warnings
from typing import Dict, List, Tuple, Sequence, Optional

import pandas as pd
from ase.io import read as ase_read

import torch
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

LOGGER = logging.getLogger("preprocess")


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
        csv_paths: Sequence[str],
        cif_path: str,
        entity_key: str,
        split_keys=None,
        num_workers: int = 0,
        rebuild_lut: bool = False,   # kept for CLI compatibility (unused)
        log_every: int = 5,
        backend: str = "thread",
        cif_suffixes: Optional[str] = None,
    ):
        """
        Loader:
          - Multiple CSVs are merged horizontally.
          - All CSVs must have the same length.
          - The first column in each CSV must be `entity_key` and match row-wise.
          - CIF path is a directory containing <ID>[SUFFIX].cif files, where SUFFIX
            is optional and may be passed via `cif_suffixes`.
        """

        self.entity_key = entity_key
        self.cif_suffixes= cif_suffixes

        # Validate CSV paths
        if not csv_paths:
            raise ValueError("At least one CSV path must be provided.")
        self.csv_paths = list(csv_paths)
        for p in self.csv_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"CSV file not found: {p}")

        # Validate CIF directory
        if not os.path.isdir(cif_path):
            raise FileNotFoundError(f"CIF folder not found: {cif_path}")
        self.cif_path = cif_path

        # Load and merge CSVs
        dfs: List[pd.DataFrame] = [pd.read_csv(p) for p in self.csv_paths]

        # Check lengths
        n_rows = len(dfs[0])
        for i, df in enumerate(dfs[1:], start=1):
            if len(df) != n_rows:
                raise ValueError(
                    f"CSV length mismatch between {self.csv_paths[0]} (n={n_rows}) "
                    f"and {self.csv_paths[i]} (n={len(df)})."
                )

        # Check entity_key presence, position, and row-wise equality
        for i, df in enumerate(dfs):
            if self.entity_key not in df.columns:
                raise KeyError(f"CSV {self.csv_paths[i]} does not contain entity_key column '{self.entity_key}'.")
            if df.columns[0] != self.entity_key:
                raise ValueError(
                    f"CSV {self.csv_paths[i]}: first column must be '{self.entity_key}', "
                    f"got '{df.columns[0]}' instead."
                )

        base_ids = dfs[0][self.entity_key].astype(str).tolist()
        for i, df in enumerate(dfs[1:], start=1):
            ids = df[self.entity_key].astype(str).tolist()
            if base_ids != ids:
                raise ValueError(
                    f"Entity order/values differ between {self.csv_paths[0]} and {self.csv_paths[i]} "
                    f"for column '{self.entity_key}'."
                )

        # Merge all dataframes horizontally, avoiding duplicate feature columns
        merged = dfs[0].copy()
        for i, df in enumerate(dfs[1:], start=1):
            new_cols = [c for c in df.columns if c != self.entity_key]
            dup = set(new_cols) & set(merged.columns)
            if dup:
                raise ValueError(
                    f"Duplicate feature columns across CSVs (excluding '{self.entity_key}'): {sorted(dup)}"
                )
            merged = pd.concat([merged, df[new_cols]], axis=1)

        self.df = merged
        self.log_every = max(1, int(log_every))
        self.backend = backend if backend in ("thread", "process") else "thread"

        # Base properties mapping
        self.properties: Dict[str, str] = {
            "Qst_CO2_kJmol-1": "qst_co2",
            "Qst_CO2_kJmol-1_std": "qst_co2_std",
            "KH_CO2_molkgPa-1": "kh_co2",
            "KH_CO2_molkgPa-1_std": "kh_co2_std",
            "Qst_H2O_kJmol-1": "qst_h2o",
            "Qst_H2O_kJmol-1_std": "qst_h2o_std",
            "KH_H2O_molkgPa-1": "kh_h2o",
            "KH_H2O_molkgPa-1_std": "kh_h2o_std",
            "Qst_N2_kJmol-1": "qst_n2",
            "Qst_N2_kJmol-1_std": "qst_n2",
            "KH_N2_molkgPa-1": "kh_n2",
            "KH_N2_molkgPa-1_std": "kh_n2_std",
            "uptake_CO2_0.0004bar_mmolg-1": "co2_uptake",
            "uptake_CO2_0.0004bar_mmolg-1_std": "co2_uptake_std",
            "uptake_N2_0.78bar_mmolg-1": "n2_uptake",
            "uptake_N2_0.78bar_mmolg-1_std": "n2_uptake_std",
            "selectivity_KH_CO2/KH_H2O": "selectivity_co2_h2o",
            "Selectivity_KH_CO2/KH_N2": "selectivity_co2_n2",
            "diff_Qst_CO2_Qst_H2O": "diff_qst_co2_h2o",
            "diff_Qst_CO2_Qst_N2": "diff_qst_co2_n2",
            # Geometric properties which are used as features and NOT as labels.
            "Di_LCD_Angs": "lcd",
            "Df_PLD_Angs": "pld",
            "Dif_GCD_Angs": "gcd",
            "Unitcell_volume_Angs3": "unitcell_volume",
            "Density_g_cm3": "density",
            "ASA_m2_g-1": "asa",
            "AV_cm3_g-1": "av",
            "NAV_cm3_g-1": "nav",
        }

        # Extend mapping: any additional CSV columns become properties with their original names
        for col in self.df.columns:
            if col == self.entity_key:
                continue
            if col not in self.properties:
                self.properties[col] = col

        # Index all .cif files once (case-insensitive), keyed by:
        #   - the full stem (without ".cif")
        #   - and, if it ends with any suffix in self.cif_suffixes,
        #     also by the stem with that suffix stripped
        file_index: Dict[str, str] = {}
        # sort suffixes longest-first to avoid partial overlaps
        sorted_suffixes = [cif_suffixes] if cif_suffixes is not None else []

        for fn in os.listdir(self.cif_path):
            if fn.lower().endswith(".cif"):
                stem = fn[:-4]  # remove '.cif'
                candidates = {stem.lower()}  # full stem key
                for suf in sorted_suffixes:
                    if stem.endswith(suf):
                        base = stem[: -len(suf)]
                        candidates.add(base.lower())
                for key in candidates:
                    if key in file_index and file_index[key] != fn:
                        raise ValueError(
                            f"Ambiguous CIF mapping for key '{key}': "
                            f"{file_index[key]} vs {fn}. "
                            f"Check your cif_suffixes or CIF directory."
                        )
                    file_index[key] = fn

        # Build ID -> filename map (strict: require a matching file)
        unique_ids: List[str] = self.df[self.entity_key].astype(str).tolist()
        self._id2file: Dict[str, str] = {}
        missing: List[str] = []

        for raw in set(unique_ids):
            key_lower = raw.strip().lower()
            hit = file_index.get(key_lower)
            if hit is None:
                missing.append(raw)
            else:
                self._id2file[key_lower] = hit

        if missing:
            examples = ", ".join(missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} CSV IDs have no matching '<ID>[SUFFIX].cif' in {self.cif_path}. "
                f"Examples: {examples}. "
                f"If your CIFs contain suffixes (e.g. '_mepoml'), pass them via --cif_suffixes."
            )

        if split_keys is None:
            self.split_keys = list(range(len(self.df)))
        else:
            self.split_keys = list(split_keys)

        self.num_workers = max(0, int(num_workers))

    def __len__(self):
        return len(self.split_keys)

    def _lookup_cif_path(self, entity_id_raw: str) -> str:
        key = entity_id_raw.strip().lower()
        try:
            fn = self._id2file[key]
        except KeyError:
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
                    row = self.df.iloc[idx]
                    entity_id = str(row[self.entity_key])
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
    parser.add_argument(
        "--csv_paths",
        required=True,
        nargs="+",
        help="One or more CSV files (same length, same first column = entity_key).",
    )
    parser.add_argument(
        "--cif_path",
        required=True,
        help="Directory containing CIF files.",
    )
    parser.add_argument("--entity_key", required=True)
    parser.add_argument("--split_keys", type=int, nargs="+", default=None)
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--backend", type=str, default="thread", choices=["thread", "process"])
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--rebuild_lut", action="store_true")  # accepted but unused
    parser.add_argument(
        "--cif_suffixes",
        type=str,
        default=None,
        help=(
            "Optional list of suffixes that may appear in CIF filenames after the entity_key, "
            "e.g. '_mepoml' or '-geo_opt_odac_mepoml'. These are stripped for matching."
        ),
    )

    args = parser.parse_args()
    _configure_logging(args.log_level)

    LOGGER.info("Initializing DataLoader...")
    dl = DataLoader(
        csv_paths=args.csv_paths,
        cif_path=args.cif_path,
        entity_key=args.entity_key,
        split_keys=args.split_keys,
        num_workers=args.num_workers,
        log_every=args.log_every,
        backend=args.backend,
        rebuild_lut=args.rebuild_lut,
        cif_suffixes=args.cif_suffixes,
    )

    count = 0
    for _, _ in dl:
        count += 1
    LOGGER.info("Finished preprocessing %d items.", count)
