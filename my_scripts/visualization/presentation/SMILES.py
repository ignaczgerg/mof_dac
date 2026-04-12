from rdkit import Chem
from rdkit.Chem import Draw
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
import numpy as np


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]



# Define the SMILES string
# AKEQUE
smiles = "[Cu].n1ccc(cc1)c1ccncc1"

# Convert SMILES to a molecule
mol = Chem.MolFromSmiles(smiles)


N = mol.GetNumAtoms()
M = mol.GetNumBonds()

type_idx = []
chirality_idx = []
atomic_number = []

for atom in mol.GetAtoms():
    type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
    chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
    atomic_number.append(atom.GetAtomicNum())


x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
x = torch.cat([x1, x2], dim=-1)

row, col, edge_feat = [], [], []
for bond in mol.GetBonds():
    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    row += [start, end]
    col += [end, start]
    edge_feat.append([
        BOND_LIST.index(bond.GetBondType()),
        BONDDIR_LIST.index(bond.GetBondDir())
    ])
    edge_feat.append([
        BOND_LIST.index(bond.GetBondType()),
        BONDDIR_LIST.index(bond.GetBondDir())
    ])

edge_index = torch.tensor([row, col], dtype=torch.long)
edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)


# Save the molecule as an image
Draw.MolToFile(mol, "AKEQUE.png", size=(300, 300))

print("Molecule image saved as 'molecule.png'")
