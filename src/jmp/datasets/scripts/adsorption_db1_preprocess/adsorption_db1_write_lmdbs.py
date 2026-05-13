"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import ase.io
import lmdb
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .adsorption_db1_dataloader import Dataloader

# from ocpmodels.preprocessing import AtomsToGraphs


def write_images_to_lmdb(mp_arg):
    db_path, data_loader, pid, args = mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    counter = 0
    pbar = tqdm(
        total=len(data_loader),
        position=pid,
        desc="Preprocessing data into LMDBs",
    )
    for idx, data_object in data_loader:
                

        txn = db.begin(write=True)
        txn.put(
            f"{counter}".encode("ascii"),
            pickle.dumps(data_object, protocol=-1),
        )
        txn.commit()
        counter += 1
        pbar.update(1)

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(counter, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return None, counter


def main(args):
        # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # Initialize lmdb paths
    db_paths = [
        os.path.join(args.out_path, "data.%04d.lmdb" % i)
        for i in range(args.num_workers)
    ]
    
    
    with open(args.split_keys, 'rb') as f:
        split_idx = pickle.load(f) 
        
    split_idx = np.array_split(
        np.array(split_idx), args.num_workers)
    data_loaders = [Dataloader(
            dataset_path=args.data_path,
            split_keys=split_idx[i],
        ) for i in range(args.num_workers)]
    
    pool = mp.Pool(args.num_workers)
    mp_args = [
        (
            db_paths[i],
            data_loaders[i],
            i, #pid
            args,
        )
        for i in range(args.num_workers)
    ]
    op = list(zip(*pool.imap(write_images_to_lmdb, mp_args)))
    # sampled_ids, idx = list(op[0]), list(op[1])

    # # Log sampled image, trajectory trace
    # for j, i in enumerate(range(args.num_workers)):
    #     ids_log = open(os.path.join(args.out_path, "data_log.%04d.txt" % i), "w")
    #     ids_log.writelines(sampled_ids[j])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", help="Path to dataset directory")
    parser.add_argument(
        "--out-path",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--split_keys",
        help="Path to split keys pickle file",
    )
    """parser.add_argument(
        "--get-edges",
        action="store_true",
        help="Store edge indices in LMDB, ~10x storage requirement. Default: compute edge indices on-the-fly.",
    )"""
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    """parser.add_argument(
        "--ref-energy", action="store_true", help="Subtract reference energies"
    )"""
    # parser.add_argument(
    #     "--test-data",
    #     action="store_true",
    #     help="Is data being processed test data?",
    # )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
