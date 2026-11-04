"""Find ConSolv solvent indices (solv_species) for solvents given as SMILES.

ConSolv's model prepends an all-zero row to descriptor.csv, so
solv_species = (row in descriptor.csv) + 1. Rows are matched by recomputing the
same 15 RDKit descriptors that ConSolv's dataset/descriptor.py uses.

    python scripts/solvent_index.py water=O dmso=CS(C)=O acetone=CC(C)=O
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

CSV = Path(__file__).resolve().parents[1] / "upstream" / "descriptor.csv"  # from scripts/fetch_upstream.py

table = pd.read_csv(CSV, index_col=0)
calc = MolecularDescriptorCalculator(list(table.columns))

for arg in sys.argv[1:]:
    name, smi = arg.split("=", 1)
    d = np.array(calc.CalcDescriptors(Chem.MolFromSmiles(smi)))
    err = np.abs(table.values - d).max(axis=1)
    row = int(np.argmin(err))
    print(f"{name:10s} {smi:12s} row={row:4d} solv_species={row + 1:4d} max|diff|={err[row]:.2e}")
