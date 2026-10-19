"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# pylint: disable=stop-iteration-return

import os
import argparse
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data

from ase import Atoms
from ase.io import cif
# from ase.calculators.singlepoint import SinglePointCalculator as SPCalc
# from ase.units import Hartree

DB2=True

class Dataloader:
    """
    Can iterate through 5419_MOFCIFs directory####

    dataset_path: path to data
    split_keys: path to split indices
    """

    def __init__(self, dataset_path, split_keys):
        if not DB2:
            self.cifs_path = os.path.join(dataset_path, '5419_MOFCIFs')
            self.csv_path = os.path.join(dataset_path, '5419_QMOF_DATABASE.csv')
        else:
            self.cifs_path = os.path.join(dataset_path, '7541_MOFs_CIFs') 
            self.csv_path = os.path.join(dataset_path, '7541_MOFs.csv')
        self.df = pd.read_csv(self.csv_path, encoding='ISO-8859-1')
        self.split_keys = split_keys
        
    def __len__(self):
        return len(self.split_keys)

    def __iter__(self):
        for idx in self.split_keys:
            mof_id = self.df.loc[idx, 'MOF ID']
            mof_info = self.df.loc[idx]
            atoms = Atoms(list(cif.read_cif(
                os.path.join(self.cifs_path, f'{mof_id}.cif'), 
                index=0, store_tags=True)))
            atoms.set_tags(2 * np.ones(atoms.get_positions().shape[0]))
            
            
            data_object = Data(
                pos=torch.Tensor(atoms.get_positions()),
                cell=torch.Tensor(atoms.get_cell().array),
                atomic_numbers=torch.LongTensor(atoms.get_atomic_numbers()),
                natoms=atoms.get_positions().shape[0],
                tags=torch.LongTensor(atoms.get_tags()),
                pbc=torch.Tensor(atoms.pbc),
                idx=idx,
                id=mof_id,
                sid=mof_id,
                lcd=mof_info['LCD  (Å)'],
                pld=mof_info['PLD (Å)'],
                gcd=mof_info['GCD (Å)'],
                unitcell_volume=mof_info['Unitcell_volume (Å^3)'],
                oms=mof_info['OMS'],
                qst_co2=mof_info['Qst_CO2_kJmol-1'],
                qst_h2o=mof_info['Qst_H2O_kJmol-1'],
                qst_n2=mof_info['Qst_N2_kJmol-1'],
                qst_co2_h2o=mof_info['?Qst(CO2-H2O)_kJmol-1'],
                qst_co2_n2=mof_info['?Qst(CO2-N2)_kJmol-1'],
                metals=mof_info['Unique Metal Entities'].split(', '),
                selectivity_co2_n2=(mof_info['Selectivity_CO2_N2(Kh_CO2/Kh_N2)']\
                                   if not DB2 else None),
            )
           
            
            
            yield idx, data_object
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        help="Path to the directory containing the csv file.",
        required=True,
    )
    
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    split_keys = range(10)
    dataloader = Dataloader(dataset_path, split_keys)
    for idx, data_object in dataloader:
        print(data_object)
        break
    print('Done')
