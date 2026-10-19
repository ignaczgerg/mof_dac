import os
import argparse
import pickle
import logging
import bisect
import warnings
from typing import Dict, List, Tuple

import pandas as pd
from ase.io import read as ase_read

import torch
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


LOGGER = logging.getLogger("hmof_preprocess")


def _apply_warning_filters():
    """Silence hot ASE warnings that add overhead and clutter logs."""
    warnings.filterwarnings(
        "ignore",
        message=r".*crystal system .* is not interpreted for space group 1.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"scaled_positions \d+ and \d+ are equivalent",
    )


def _prefix_range(sorted_keys: List[str], prefix: str) -> List[str]:
    """Return sublist of sorted_keys that start with `prefix` using bisection."""
    left = bisect.bisect_left(sorted_keys, prefix)
    right = bisect.bisect_left(sorted_keys, prefix + "\uffff")
    return sorted_keys[left:right]


def map_entity_to_cif(
    dataframe: pd.DataFrame,
    entity_col: str,
    cif_dir: str,
    save_dir: str,
    get_lut: bool = False,
) -> Dict[str, str]:
    """
    Build a mapping from entity name -> CIF file in `cif_dir`.

    Backward compatible with the original signature:
    - `save_dir` may be a directory OR a full .pkl path.

    Returns the LUT if `get_lut` is True, else saves it and returns nothing.
    """
    # Resolve save path (directory or file path accepted)
    if save_dir and (save_dir.endswith(".pkl") or os.path.splitext(save_dir)[1]):
        save_path = save_dir
    else:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "entity_cif_map.pkl")

    files = [f for f in os.listdir(cif_dir) if f.lower().endswith(".cif")]
    basenames = [os.path.splitext(f)[0] for f in files]
    sorted_basenames = sorted(basenames)
    base_to_file = {bn: fn for bn, fn in zip(basenames, files)}

    entities = dataframe[entity_col].astype(str).unique().tolist()
    LOGGER.info(
        "Mapping %d entities to %d CIF files in '%s'...",
        len(entities), len(files), cif_dir,
    )

    lut: Dict[str, str] = {}
    no_match, multiple_match = [], []

    for i, name in enumerate(entities, 1):
        if i % 1000 == 0 or i == len(entities):
            LOGGER.debug(" map_entity_to_cif progress: %d/%d", i, len(entities))

        # Fast path: exact basename match
        if name in base_to_file:
            lut[name] = base_to_file[name]
            continue

        # Prefix search via bisection
        candidate_basenames = _prefix_range(sorted_basenames, name)
        if len(candidate_basenames) == 1:
            lut[name] = base_to_file[candidate_basenames[0]]
        elif len(candidate_basenames) > 1:
            multiple_match.append((name, [base_to_file[bn] for bn in candidate_basenames]))
        else:
            no_match.append(name)

    if no_match:
        raise FileNotFoundError(
            f"Entities not found in CIFs: {no_match[:10]}{' ...' if len(no_match)>10 else ''}"
        )
    if multiple_match:
        raise ValueError(
            f"Entities with multiple CIFs: {multiple_match[:5]}{' ...' if len(multiple_match)>5 else ''}"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(lut, f)
    LOGGER.info("Saved entity→CIF map to %s", save_path)

    if get_lut:
        return lut
    return {}


# ---------- Process-mode worker: parse CIF only, return arrays ----------

def _read_cif_worker(task):
    """
    Worker function for ProcessPoolExecutor.

    Args:
        task: tuple (idx, cif_path)
    Returns:
        (idx, pos, cell, atomic_numbers, pbc)
        or (idx, exc) if an exception occurred
    """
    _apply_warning_filters()  # ensure filters are active in child
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
        rebuild_lut: bool = False,
        log_every: int = 200,
        backend: str = "thread",  # NEW: 'thread' (default) or 'process'
    ):
        """
        base_folder: dataset root folder
        split_keys: iterable of positional row indices (ints). If None → all rows.
        entity_key: column name in CSV with the MOF/structure identifier
        num_workers: parallelism degree; 0/1 = no parallelism
        rebuild_lut: force rebuild of entity→CIF LUT even if cache exists
        log_every: log a progress line every N items during iteration
        backend: 'thread' (I/O bound) or 'process' (CPU-bound CIF parsing)
        """
        self.entity_key = entity_key
        self.cif_path = os.path.join(base_folder, "raw", "MOF_database")
        self.csv_path = os.path.join(base_folder, "raw", "all_MOFs_screening_data_cleaned.csv")
        self.metadata = os.path.join(base_folder, "metadata", "entity_cif_map.pkl")
        self.df = pd.read_csv(self.csv_path)
        self.log_every = max(1, int(log_every))
        self.backend = backend if backend in ("thread", "process") else "thread"

        self.properties = {
            "CO2_uptake_P0.15bar_T298K [mmol/g]": "uptake_co2",
            "heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]": "qst_co2",
            "excess_CO2_uptake_P0.15bar_T298K [mmol/g]": "excess_co2",
            "CO2_binary_uptake_P0.15bar_T298K [mmol/g]": "uptake_binary_co2",
            "heat_adsorption_CO2_binary_P0.15bar_T298K [kcal/mol]": "qst_binary_co2",
            "excess_CO2_binary_uptake_P0.15bar_T298K [mmol/g]": "excess_binary_co2",
            "N2_binary_uptake_P0.85bar_T298K [mmol/g]": "uptake_binary_n2",
            "heat_adsorption_N2_binary_P0.85bar_T298K [kcal/mol]": "qst_binary_n2",
            "excess_N2_binary_uptake_P0.85bar_T298K [mmol/g]": "excess_binary_n2",
            "CO2/N2_selectivity": "selectivity_co2_n2",
        }

        # Load or (re)build LUT
        if os.path.exists(self.metadata) and not rebuild_lut:
            with open(self.metadata, "rb") as f:
                self.entity_map = pickle.load(f)
            LOGGER.info("Loaded cached entity→CIF map from %s", self.metadata)
        else:
            self.entity_map = map_entity_to_cif(
                dataframe=self.df,
                entity_col=self.entity_key,
                cif_dir=self.cif_path,
                save_dir=self.metadata, 
                get_lut=True,
            )

        if split_keys is None:
            self.split_keys = list(range(len(self.df)))
        else:
            self.split_keys = list(split_keys)

        self.num_workers = max(0, int(num_workers))

    def __len__(self):
        return len(self.split_keys)

    def _build_one(self, pos_idx: int) -> Tuple[int, Data]:
        """Thread path: read CIF and build Data in this process."""
        entity_info = self.df.iloc[pos_idx]
        entity_id = str(entity_info[self.entity_key])
        atoms = ase_read(
            os.path.join(self.cif_path, self.entity_map[entity_id]),
            index=0,
            format="cif",
        )
        data_object = Data(
            pos=torch.tensor(atoms.get_positions(), dtype=torch.float32),
            cell=torch.tensor(atoms.cell.array, dtype=torch.float32),
            atomic_numbers=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
            natoms=int(len(atoms)),
            pbc=torch.tensor(atoms.pbc, dtype=torch.bool),
            idx=pos_idx,
            id=entity_id,
            sid=entity_id,
            **{short: entity_info[csv_col] for csv_col, short in self.properties.items()},
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
                    for k, (pos_idx, data_object) in enumerate(
                        ex.map(self._build_one, self.split_keys), 1
                    ):
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
            # build task stream: (idx, cif_path)
            def _tasks():
                for idx in self.split_keys:
                    entity_id = str(self.df.iloc[idx][self.entity_key])
                    cif_path = os.path.join(self.cif_path, self.entity_map[entity_id])
                    yield (idx, cif_path)

            with ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_apply_warning_filters,
            ) as ex:
                for k, result in enumerate(ex.map(_read_cif_worker, _tasks()), 1):
                    idx, payload = result[0], result[1:]
                    # Error handling
                    if len(payload) == 1 and isinstance(payload[0], Exception):
                        LOGGER.warning("Skipping idx=%s due to CIF read error: %s", idx, payload[0])
                        continue

                    pos, cell, zs, pbc = payload
                    entity_info = self.df.iloc[idx]
                    entity_id = str(entity_info[self.entity_key])

                    data_object = Data(
                        pos=torch.tensor(pos, dtype=torch.float32),
                        cell=torch.tensor(cell, dtype=torch.float32),
                        atomic_numbers=torch.tensor(zs, dtype=torch.long),
                        natoms=int(len(zs)),
                        pbc=torch.tensor(pbc, dtype=torch.bool),
                        idx=idx,
                        id=entity_id,
                        sid=entity_id,
                        **{
                            short: entity_info[csv_col]
                            for csv_col, short in self.properties.items()
                        },
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
    _apply_warning_filters()  # also install in the parent process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--entity_key", required=True,
        help="Column name in the CSV that identifies the entity (e.g., MOF name).",
    )
    parser.add_argument(
        "--split_keys", type=int, nargs="+",
        help="List of positional row indices to iterate. Omit to use all rows.",
        default=None,
    )
    parser.add_argument(
        "--num_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallelism degree (>=2 enables parallelism).",
    )
    parser.add_argument(
        "--backend", type=str, default="thread", choices=["thread", "process"],
        help="Parallel backend for CIF parsing (default: thread).",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR.",
    )
    parser.add_argument(
        "--log_every", type=int, default=200,
        help="Log progress every N items.",
    )
    parser.add_argument(
        "--rebuild_lut", action="store_true",
        help="Force rebuilding the entity→CIF LUT cache.",
    )

    args = parser.parse_args()
    _configure_logging(args.log_level)

    LOGGER.info("Initializing DataLoader...")
    dl = DataLoader(
        base_folder=args.dataset_path,
        entity_key=args.entity_key,
        split_keys=args.split_keys,
        num_workers=args.num_workers,
        rebuild_lut=args.rebuild_lut,
        log_every=args.log_every,
        backend=args.backend,  
    )

    count = 0
    for _, _ in dl:
        count += 1
    LOGGER.info("Finished preprocessing %d items.", count)
