"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# pylint: disable=stop-iteration-return

import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data

from ase import Atoms
from ase.io import cif
# from ase.calculators.singlepoint import SinglePointCalculator as SPCalc
# from ase.units import Hartree

MOF_DB_1=False
MOF_DB_2=True

assert MOF_DB_1 != MOF_DB_2, "Set only one of MOF_DB_1 or MOF_DB_2 to True"

def map_mof_to_cif(df, CIFs_dir):
    '''function to map MOF ID to CIF file for MOF-DB-1.0'''
    mof_cif_map = {}
    no_matching = []
    more_than_one = []
    
    cif_files = set(os.listdir(CIFs_dir))
    
    for mof_name in df['MOF ID'].values:
        
        matching = [f for f in cif_files if f.startswith(mof_name)]
        if len(matching) == 0:
            no_matching.append(mof_name)
            continue
        elif len(matching) > 1:
            for cif_file in matching:
                if cif_file.split('-geo')[0] == mof_name:
                    mof_cif_map[mof_name] = cif_file
                    break
            else:
                more_than_one.append(mof_name)
        else: # this handles cif files where '-geo' is not present (e.g. CALF-20_111_mepoml.cif)
            mof_cif_map[mof_name] = matching[0]
    
    assert len(no_matching) == 0, f"MOF IDs not found in CIFs: {no_matching}"
    assert len(more_than_one) == 0, f"MOF IDs with more than one CIF: {more_than_one}"
    
    return mof_cif_map 

class Dataloader:
    """
    Can iterate through MOFs directory given a CSV file
    """

    def __init__(self, dataset_path, split_keys):
        """
        Initializes the dataset loader.

        Args:
            dataset_path (str): Path to "dataset/raw".
            split_keys (str): Path to the split keys file \
            (e.g., "dataset/split_keys/{split}_keys.pkl").
        """
        if MOF_DB_1:
            self.cifs_path = os.path.join(dataset_path, 'MOF-DB-1.0_CIFs_28012025')
            self.csv_path = os.path.join(dataset_path, 'MOF-DB-1.0_DATABASE_28012025.csv')
            self.mof_cif_map_path = os.path.join(dataset_path, '../metadata/mof_cif_map.pkl')
            
        elif MOF_DB_2:
            self.cifs_path = os.path.join(dataset_path, 'MOF_DB_2_0_hMOF_CIFs_18042025') 
            self.csv_path = os.path.join(dataset_path, 'MOF_DB_2_0_18042025.csv')
        
        else: # to be modified to handle cof-db-1
            # self.cifs_path = os.path.join(dataset_path, '7541_MOFs_CIFs') 
            # self.csv_path = os.path.join(dataset_path, '7541_MOFs.csv')
            pass
        
        
        self.df = pd.read_csv(self.csv_path)
        # self.mof_cif_map = map_mof_to_cif(self.df, self.cifs_path)
        if MOF_DB_1:
            self.mof_id_key = 'MOF ID'
            self.properties = {
                'lcd': 'LCD ',
                'pld': 'PLD ',
                'gcd': 'GCD',
                'unitcell_volume': 'Unitcell_volume ',
                'oms': 'OMS',
                # Isosteric heat of adsorption values
                'qst_co2': 'Qst_CO2(kJ/mol)',
                'qst_h2o': 'Qst_H2O(kJ/mol)',
                'qst_n2': 'Qst_N2(kJ/mol)',
                # Henry's law constant values
                'kh_co2': 'KH_CO2(molkg-1Pa-1)',
                'kh_h2o': 'KH_H2O(molkg-1Pa-1)',
                'kh_n2': 'KH_N2(molkg-1Pa-1)',
                # Selectivity values
                'selectivity_co2_h2o': 'selectivity_KH_CO2/KH_H2O',
                'selectivity_co2_n2': 'Selectivity_KH_CO2/KH_N2',
                # Uptake values
                'co2_uptake': 'CO2_uptake_400ppm_mmolg-1',
                'n2_uptake': 'N2_uptake_400ppm_mmolg-1',
            }
            with open(self.mof_cif_map_path, "rb") as f:
                self._mof_cif_map = pickle.load(f)
                self.mof_cif_map = (
                    lambda mof_id: self._mof_cif_map[mof_id]
                )
        if MOF_DB_2:
            self.mof_id_key = 'HMOF_ID'
            self.properties = {
                'lcd': 'LCD',
                'pld': 'PLD',
                'gcd': 'GLD', #TODO: check what this discriptor is
                'unitcell_volume': 'UC_volume',
                'oms': 'Has OMS',
                # Isosteric heat of adsorption values
                'qst_co2': 'Qst_CO2_kJmol-1',
                'qst_h2o': 'Qst_H2O_kJmol-1',
                # Henry's law constant values
                'kh_co2': 'Kh_CO2_molkg-1Pa-1',
                'kh_h2o': 'Kh_H2O_molkg-1Pa-1',
                # Selectivity values
                'selectivity_co2_h2o': 'Selectivity_CO2/H2O',
                # Uptake values
                'co2_uptake': 'CO2_uptake (mmol/g)',
            }
            self.mof_cif_map = (
                lambda mof_id: os.path.join(
                    self.cifs_path, f"{mof_id}.cif")
            )
        self.split_keys = split_keys
        
    def __len__(self):
        return len(self.split_keys)

    def __iter__(self):
        for idx in self.split_keys:
            mof_id = self.df.loc[idx, self.mof_id_key]
            mof_info = self.df.loc[idx]
            atoms = cif.read_cif(
                os.path.join(self.cifs_path, self.mof_cif_map(mof_id)), 
                index=0, store_tags=True)
            
            
            data_object = Data(
                pos=torch.Tensor(atoms.get_positions()),
                cell=torch.Tensor(atoms.get_cell().array),
                atomic_numbers=torch.LongTensor(atoms.get_atomic_numbers()),
                natoms=atoms.get_positions().shape[0],
                pbc=torch.Tensor(atoms.pbc),
                idx=idx,
                id=mof_id,
                sid=mof_id,
                **{prop: mof_info[self.properties[prop]] for prop in self.properties},
                metals=mof_info['Unique Metal Entities'].split(', '),
            )  
           
            
            
            yield idx, data_object
            
            
class AIGeneratedDataloader:
    """
    DataLoader for AI-generated MOFs.
    Can iterate through a directory of CIF files
    """

    def __init__(self, cifs_path, split_keys):
        """
        Initializes the dataset loader.

        Args:
            cifs_path (str): Path to the directory containing CIF files.
            split_keys (np.array): Array of indices for splitting the dataset.
        """
        self.cifs_path = cifs_path
        self.split_keys = split_keys
        self.cifs_files_list = os.listdir(cifs_path)

    def __len__(self):
        return len(self.split_keys)

    def __iter__(self):
        for idx in self.split_keys:
            file = self.cifs_files_list[idx]
            mof_id = file.split('.cif')[0]
            atoms = cif.read_cif(self.cifs_path + file, 
                index=0, store_tags=True)
            
            data_object = Data(
                pos=torch.Tensor(atoms.get_positions()),
                cell=torch.Tensor(atoms.get_cell().array),
                atomic_numbers=torch.LongTensor(atoms.get_atomic_numbers()),
                natoms=atoms.get_positions().shape[0],
                pbc=torch.Tensor(atoms.pbc),
                idx=idx,
                id=mof_id,
                sid=mof_id,
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
    # args = argparse.Namespace(
    #     dataset_path='/home/hani/dev/datasets/adsorption-mof-db2/raw' 
    # )
    
    dataset_path = args.dataset_path
    split_keys = range(10)
    if MOF_DB_1:
        cifs_path = os.path.join(dataset_path, 'MOF-DB-1.0_CIFs_28012025')
        csv_path = os.path.join(dataset_path, 'MOF-DB-1.0_DATABASE_28012025.csv')
        mof_cif_map_path = os.path.join(dataset_path, 'metadata', 'mof_cif_map.pkl')
        mof_cif_map = map_mof_to_cif(pd.read_csv(csv_path), cifs_path)
        print("the type here is", type(mof_cif_map))
    
    
        with open(os.path.join(dataset_path, '../metadata/mof_cif_map.json'), "w") as f:
            json.dump(mof_cif_map, f, indent=4)
        with open(os.path.join(dataset_path, '../metadata/mof_cif_map.pkl'), "wb") as f:
            pickle.dump(mof_cif_map, f)
        print("Successfully dumped mof_cif_map.{pkl, json}")
    else: # to be modified to handle cof-db-1
        # self.cifs_path = os.path.join(dataset_path, '7541_MOFs_CIFs') 
        # self.csv_path = os.path.join(dataset_path, '7541_MOFs.csv')
        pass
    # dataloader = Dataloader(dataset_path, split_keys)
    # for idx, data_object in dataloader:
    #     print(data_object)
    #     if idx == 5:
    #         break
    cifs_path = "/ibex/project/c2261/datasets/adsorption-mof-ai-generated-2_0/cif/"
    dataloader = AIGeneratedDataloader(cifs_path, split_keys)
    for idx, data_object in dataloader:
        print(data_object)
        if idx == 5:
            break
    print('Done')