import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Sequence, Optional

import lmdb
import numpy as np

from preprocess import DataLoader, _configure_logging

LOGGER = logging.getLogger("lmdb-writer")


def _derive_all_indices_from_csv(csv_path: str) -> List[int]:
    """Infer total number of rows from the CSV and return range(len)."""
    import pandas as pd
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, usecols=[0])
    return list(range(len(df)))


def _make_splits(indices, ratio=(6, 2, 2), seed=42) -> Dict[str, List[int]]:
    """Shuffle indices and split with given ratio."""
    idx = np.array(list(map(int, indices)), dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    a, b, c = ratio
    n = len(idx)
    n_train = int(n * a / (a + b + c))
    n_val = int(n * b / (a + b + c))
    train = idx[:n_train].tolist()
    val = idx[n_train:n_train + n_val].tolist()
    test = idx[n_train + n_val:].tolist()
    return {"train": train, "val": val, "test": test}


def _save_splits(out_dir: Path, splits: Dict[str, List[int]], subdir: str = "split_keys"):
    path = out_dir / subdir
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "train_keys.pkl", "wb") as f:
        pickle.dump(splits["train"], f)
    with open(path / "val_keys.pkl", "wb") as f:
        pickle.dump(splits["val"], f)
    with open(path / "test_keys.pkl", "wb") as f:
        pickle.dump(splits["test"], f)
    with open(out_dir / "splits.json", "w") as f:
        json.dump({k: {"count": len(v)} for k, v in splits.items()}, f, indent=2)
    LOGGER.info("Saved split keys under %s", path)


def _write_split_lmdb(
    split_name: str,
    keys: List[int],
    csv_paths: Sequence[str],
    cif_path: str,
    entity_key: str,
    num_workers: int,
    rebuild_lut: bool,
    log_every: int,
    out_root: Path,
    map_size_gb: float,
    commit_interval: int,
    reindex: bool = True,
    backend: str = "thread",
    cif_suffixes: Optional[str] = None,
):
    split_dir = out_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = split_dir / f"{split_name}.lmdb"

    env = lmdb.open(
        str(lmdb_path),
        map_size=int(map_size_gb * (1024 ** 3)),
        subdir=False,
        meminit=False,
        map_async=True,
        lock=True,
        readahead=False,
        writemap=False,
        max_dbs=1,
    )
    LOGGER.info("Writing %s --> %s (reindex=%s, backend=%s)", split_name, lmdb_path, reindex, backend)

    dl = DataLoader(
        csv_paths=csv_paths,
        cif_path=cif_path,
        entity_key=entity_key,
        split_keys=keys,
        num_workers=num_workers,
        rebuild_lut=rebuild_lut,
        log_every=log_every,
        backend=backend,
        cif_suffixes=cif_suffixes,
    )

    n_ok, n_err = 0, 0
    natoms_list = []
    txn = env.begin(write=True)

    for _, data in dl:
        try:
            key_bytes = f"{n_ok}".encode("ascii") if reindex else f"{int(data.idx)}".encode("ascii")
            val = pickle.dumps(data, protocol=-1)
            txn.put(key_bytes, val)
            n_ok += 1
            natoms_list.append(int(getattr(data, "natoms", len(data.atomic_numbers))))
            if n_ok % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)
        except Exception as e:
            n_err += 1
            LOGGER.warning("Skipping idx=%s due to error: %s", getattr(data, "idx", "?"), e)

    txn.put(b"length", pickle.dumps(n_ok, protocol=-1))
    txn.commit()
    env.sync()
    env.close()
    LOGGER.info("Finished %s: %d items written (%d errors).", split_name, n_ok, n_err)

    meta_path = split_dir / "metadata.npz"
    np.savez(meta_path, natoms=np.array(natoms_list, dtype=np.int32))
    LOGGER.info("Wrote %s with natoms=%d", meta_path, len(natoms_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV files (same length, same first column = entity_key).",
    )
    parser.add_argument(
        "--cif_path",
        type=str,
        required=True,
        help="Directory containing CIF files.",
    )
    parser.add_argument("--entity_key", required=True,
                        help="Column name in the CSVs that identifies the entity (e.g., MOF name).")
    parser.add_argument("--split_keys", type=int, nargs="+",
                        help="List of positional row indices to iterate. Omit to use all rows.", default=None)
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Parallelism for CIF --> Data construction (>=2 enables parallelism).")
    parser.add_argument("--backend", type=str, default="thread", choices=["thread", "process"],
                        help="Parallel backend for CIF parsing.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log progress every N items.")
    parser.add_argument("--rebuild_lut", action="store_true",
                        help="Kept for compatibility; unused.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="If set, write train/val/test LMDBs here and save split keys.")
    parser.add_argument("--map_size_gb", type=float, default=8.0,
                        help="LMDB map size per split file.")
    parser.add_argument("--commit_interval", type=int, default=1000,
                        help="LMDB commit every N items.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=str, default="6,2,2",
                        help="Train,Val,Test ratio, e.g. 8,1,1")

    parser.add_argument("--train_keys_file", type=str, default=None)
    parser.add_argument("--val_keys_file", type=str, default=None)
    parser.add_argument("--test_keys_file", type=str, default=None)

    parser.add_argument("--no_reindex", action="store_true",
                        help="If set, keep original CSV indices as LMDB keys.")

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

    # Validate CSV paths early
    for p in args.csv_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"CSV file not found: {p}")

    if not os.path.isdir(args.cif_path):
        raise FileNotFoundError(f"CIF folder not found: {args.cif_path}")

    csv_paths = args.csv_paths

    if args.out_dir is None:
        LOGGER.info("Initializing DataLoader (dry run, no LMDB writing)...")
        dl = DataLoader(
            csv_paths=csv_paths,
            cif_path=args.cif_path,
            entity_key=args.entity_key,
            split_keys=args.split_keys,
            num_workers=args.num_workers,
            rebuild_lut=args.rebuild_lut,
            log_every=args.log_every,
            backend=args.backend,
            cif_suffixes=args.cif_suffixes,
        )
        count = 0
        for _, _ in dl:
            count += 1
        LOGGER.info("Finished preprocessing %d items.", count)
        return

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.train_keys_file and args.val_keys_file and args.test_keys_file:
        with open(args.train_keys_file, "rb") as f:
            train_keys = pickle.load(f)
        with open(args.val_keys_file, "rb") as f:
            val_keys = pickle.load(f)
        with open(args.test_keys_file, "rb") as f:
            test_keys = pickle.load(f)
        splits = {"train": train_keys, "val": val_keys, "test": test_keys}
        LOGGER.info("Loaded precomputed split keys.")
    else:
        indices = _derive_all_indices_from_csv(csv_paths[0]) if args.split_keys is None else args.split_keys
        ratio = tuple(int(x) for x in args.ratio.split(","))
        splits = _make_splits(indices, ratio=ratio, seed=args.seed)
        _save_splits(out_root, splits, subdir="split_keys")

    reindex = not args.no_reindex
    for split_name in ("train", "val", "test"):
        _write_split_lmdb(
            split_name=split_name,
            keys=splits[split_name],
            csv_paths=csv_paths,
            cif_path=args.cif_path,
            entity_key=args.entity_key,
            num_workers=args.num_workers,
            rebuild_lut=args.rebuild_lut,
            log_every=args.log_every,
            out_root=out_root,
            map_size_gb=args.map_size_gb,
            commit_interval=args.commit_interval,
            reindex=reindex,
            backend=args.backend,
            cif_suffixes=args.cif_suffixes,
        )
    LOGGER.info("All splits written under %s", out_root)


if __name__ == "__main__":
    main()
