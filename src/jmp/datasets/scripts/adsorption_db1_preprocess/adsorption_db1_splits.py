"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pickle
import pandas as pd
import numpy as np



def main(args):
    df = pd.read_csv(args.input_file, encoding='ISO-8859-1')
    num_elements = len(df)
    indices = list(range(num_elements))
    np.random.seed(70)
    np.random.shuffle(indices)
    
    test_size = int(args.test_split * num_elements)
    val_size = int(args.val_split * num_elements)
    
    val_indices = indices[:val_size]
    test_indices = indices[val_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    with open(args.train_keys_output, "wb") as f:
        pickle.dump(train_indices, f)
    with open(args.test_keys_output, "wb") as f:
        pickle.dump(test_indices, f)
    with open(args.val_keys_output, "wb") as f:
        pickle.dump(val_indices, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split ADSORPTION-DB1 dataset into train, test, and validation sets."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input ADSORPTION-DB1 csv file.",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Train split (default: 0.8).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split (default: 0.1).",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Train split (default: 0.1).",
    )
    parser.add_argument(
        "--train_keys_output",
        type=str,
        required=True,
        help="Path to save the train keys pickle file.",
    )
    parser.add_argument(
        "--test_keys_output",
        type=str,
        required=True,
        help="Path to save the test keys pickle file.",
    )
    parser.add_argument(
        "--val_keys_output",
        type=str,
        required=True,
        help="Path to save the validation keys pickle file.",
    )

    args = parser.parse_args()
    main(args)
