"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pickle
from functools import cache
from pathlib import Path

import multiprocess as mp
import numpy as np
import torch
from jmp.datasets.pretrain_lmdb import PretrainDatasetConfig, PretrainLmdbDataset
from torch_scatter import scatter
from tqdm import tqdm


# def _compute_mean_std(args: argparse.Namespace):
#     @cache
#     def dataset():
#         return PretrainLmdbDataset(
#             PretrainDatasetConfig(src=args.src, lin_ref=args.linref_path)
#         )

#     def extract_data(idx):
#         data = dataset()[idx]
#         y = data.y
#         na = data.natoms
#         return (y, na)

#     pool = mp.Pool(args.num_workers)
#     indices = range(len(dataset()))

#     outputs = list(tqdm(pool.imap(extract_data, indices), total=len(indices)))

#     energies = [y for y, na in outputs]
#     num_atoms = [na for y, na in outputs]

#     energy_mean = np.mean(energies)
#     energy_std = np.std(energies)
#     avg_num_atoms = np.mean(num_atoms)

#     print(
#         f"energy_mean: {energy_mean}, energy_std: {energy_std}, average number of atoms: {avg_num_atoms}"
#     )

#     with open(args.out_path, "wb") as f:
#         pickle.dump(
#             {
#                 "energy_mean": energy_mean,
#                 "energy_std": energy_std,
#                 "avg_num_atoms": avg_num_atoms,
#             },
#             f,
#         )
        
def _compute_mean_std(args: argparse.Namespace):
    @cache
    def dataset():
        return PretrainLmdbDataset(
            PretrainDatasetConfig(src=args.src, lin_ref=args.linref_path)
        )

    def extract_data(idx):
        data = dataset()[idx]
        y_co2 = data.qst_co2
        y_h2o = data.qst_h2o
        y_n2  = data.qst_n2
        na = data.natoms
        return (na, y_co2, y_h2o, y_n2)

    pool = mp.Pool(args.num_workers)
    indices = range(len(dataset()))

    outputs = list(tqdm(pool.imap(extract_data, indices), total=len(indices)))

    num_atoms = [x[0] for x in outputs]
    y_co2 = [x[1] for x in outputs]
    y_h2o = [x[2] for x in outputs]
    y_n2  = [x[3] for x in outputs]


    y_co2_mean = np.mean(y_co2)
    y_h20_mean = np.mean(y_h20)
    y_n2_mean  = np.mean(y_n2)
    y_co2_std  = np.std(y_co2)
    y_h20_std  = np.std(y_h20)
    y_n2_std   = np.std(y_n2)
    avg_num_atoms = np.mean(num_atoms)

    print(
        f"energy_mean: co2: {y_co2_mean},\th2o: {y_h2o_mean}\tn2: {y_n2_mean}\n", 
        f"energy_std:v co2{energy_std}, average number of atoms: {avg_num_atoms}"
    )

    with open(args.out_path, "wb") as f:
        pickle.dump(
            {
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "avg_num_atoms": avg_num_atoms,
            },
            f,
        )


def _linref(args: argparse.Namespace):
    @cache
    def dataset():
        return PretrainLmdbDataset(PretrainDatasetConfig(src=args.src))
    
    dataset()
    print(f"{len(dataset())=}")
    print(dataset()[0])
    
    
    def extract_data(idx):
        data = dataset()[idx]
        x = (
            scatter(
                torch.ones(data.atomic_numbers.shape[0]),
                data.atomic_numbers.long(),
                dim_size=100,
            )
            .long()
            .numpy()
        )
        y_co2 = data.qst_co2
        y_h2o = data.qst_h2o
        y_n2  = data.qst_n2
        return (x, y_co2, y_h2o, y_n2)

    pool = mp.Pool(args.num_workers)
    indices = range(len(dataset()))

    outputs = list(tqdm(pool.imap(extract_data, indices), total=len(indices)))

    features = [x[0] for x in outputs]
    targets = [x[1] for x in outputs]

    X = np.vstack(features)
    y = y_co2 = [x[1] for x in outputs]
    y_h2o = [x[2] for x in outputs]
    y_n2  = [x[3] for x in outputs]

    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    np.savez_compressed(args.out_path, coeff=coeff)
    print(f"Saved linear reference coefficients to {args.out_path}")


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subcommand")

    compute_mean_std_parser = subparsers.add_parser("compute_mean_std")
    compute_mean_std_parser.add_argument("--src", type=Path, required=True)
    compute_mean_std_parser.add_argument("--out_path", type=Path, required=True)
    compute_mean_std_parser.add_argument("--linref_path", type=Path, required=True)
    compute_mean_std_parser.add_argument("--num_workers", type=int, default=32)
    compute_mean_std_parser.set_defaults(fn=_compute_mean_std)

    linref_parser = subparsers.add_parser("linref")
    linref_parser.add_argument("--src", type=Path, required=True)
    linref_parser.add_argument("--out_path", type=Path, required=True)
    linref_parser.add_argument("--num_workers", type=int, default=32)
    linref_parser.set_defaults(fn=_linref)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
