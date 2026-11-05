"""GFN2-xTB/ALPB reference binding energies on geometries from binding_energy.py.

For each <results>/<system>_<solvent>_relaxed.xyz: GFN2/ALPB single points of the
complex, rod and wheel (each at its geometry in that complex), in that solvent.
BE = E(complex) - E(rod) - E(wheel), kcal/mol, written to <results>/gfn2_binding.csv.
Solvent names must be ALPB names (hexane, chloroform, acetone, dmf, water, ...), or gas.

Run with the mace repo's env (tblite), serially:
    ~/python_mac/mace/.venv/bin/python scripts/gfn2_binding.py results/alpb5_ft \
        --systems rot1 rot2 --solvents hexane chloroform acetone dmf water
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import scipy.sparse.csgraph as csgraph
from ase.data import covalent_radii
from ase.io import read

sys.path.insert(0, str(Path.home() / "python_mac/mace/code"))
import gfn2_data  # noqa: E402

KCAL_PER_EV = 23.0605478

p = argparse.ArgumentParser()
p.add_argument("results")
p.add_argument("--systems", nargs="+", required=True)
p.add_argument("--solvents", nargs="+", required=True)
args = p.parse_args()
res = Path(args.results)

rows = []
for system in args.systems:
    for solv in args.solvents:
        cplx = read(res / f"{system}_{solv}_relaxed.xyz")
        r = covalent_radii[cplx.numbers]
        ncomp, label = csgraph.connected_components(
            cplx.get_all_distances() < 1.2 * (r[:, None] + r[None, :]), directed=False)
        assert ncomp == 2, f"{system} {solv}: {ncomp} components"
        wheel = [c for c in range(2) if 7 not in cplx.numbers[label == c]][0]
        E = {}
        for part, atoms in [("complex", cplx), ("rod", cplx[label != wheel]), ("wheel", cplx[label == wheel])]:
            atoms = atoms.copy()
            kw = {} if solv == "gas" else {"solvation": ("alpb", solv)}  # "gas": plain GFN2
            E[part] = gfn2_data.gfn2_singlepoint(atoms, **kw)["energy"] * KCAL_PER_EV
        be = E["complex"] - E["rod"] - E["wheel"]
        print(f"{system} {solv}: GFN2/ALPB BE = {be:.2f} kcal/mol", flush=True)
        rows.append({"system": system, "solvent": solv, "BE_gfn2_alpb_kcal": be})
pd.DataFrame(rows).to_csv(res / "gfn2_binding.csv", index=False, float_format="%.3f")
