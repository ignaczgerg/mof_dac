# metadata.py

import lmdb
import pickle
import numpy as np
import os
import argparse
from torch_geometric.data import Data


def collect_natoms(lmdb_path):
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
    natoms = []
    with env.begin() as txn:
        for k, v in txn.cursor():
            if k == b"length":  # skip meta
                continue
            d: Data = pickle.loads(v)
            n = getattr(d, "natoms", getattr(d, "num_nodes", None))
            if n is None:
                n = d.atomic_numbers.shape[0]
            natoms.append(int(n))
    env.close()
    return np.array(natoms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lmdb_root",
        required=True,
        help="Root directory containing per-split LMDB subdirectories (e.g. train/val/test).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Names of split subdirectories under --lmdb_root to process.",
    )

    args = parser.parse_args()

    for split in args.splits:
        p = os.path.join(args.lmdb_root, split)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Split directory not found: {p}")
        lmdb_files = [f for f in os.listdir(p) if f.endswith(".lmdb")]
        assert len(lmdb_files) == 1, f"Expected exactly one .lmdb file in {p}, found {lmdb_files}"
        lmdb_path = os.path.join(p, lmdb_files[0])
        natoms = collect_natoms(lmdb_path)
        np.savez(os.path.join(p, "metadata.npz"), natoms=natoms)
