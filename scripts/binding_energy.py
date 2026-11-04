"""ConSolv implicit-solvent binding energies for rotaxanes.

    BE = E(complex) - E(rod) - E(wheel)

The complex is relaxed on the ConSolv surface in each solvent; rod and wheel
are single points at their geometries in the relaxed complex. They are
evaluated with the wheel translated 50 A away (far beyond the 2x5 A receptive
field) and the other component masked. Do NOT mask in place: jax-md's
neighbour-list update only rebuilds once atoms move, so an in-place mask is
silently ignored at an unmoved geometry and gives BE == 0.

Model loading reproduces ConSolv's examples/SFE/evaluation/evaluate_molecule.py
(published vacuum MACE + solvent-conditional params). ConSolv units are nm and
kJ/mol; inputs/outputs here are Angstrom xyz and kcal/mol.

    python scripts/binding_energy.py inputs/rot1.xyz inputs/rot2.xyz \
        --solvents water=1 dmso=104 acetone=7 [--params runs/<name>]
(solv_species indices from scripts/solvent_index.py)
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"  # published ConSolv files, from scripts/fetch_upstream.py

import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
import scipy.sparse.csgraph as csgraph
import tomli
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.data import covalent_radii
from ase.optimize import LBFGS
from jax_md import partition, space

import consolv_model
from chemtrain.data import graphs

KJ_PER_EV = 96.48533212
KCAL_PER_KJ = 1 / 4.184

p = argparse.ArgumentParser()
p.add_argument("xyz", nargs="+")
p.add_argument("--solvents", nargs="+", required=True, help="name=solv_species")
p.add_argument("--fmax", type=float, default=0.02, help="eV/A")
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--params", help="a runs/<name> dir from train.py --model solvent "
                                 "(default: the published ConSolv checkpoint)")
p.add_argument("--out", default=str(ROOT / "results"))
args = p.parse_args()
solvents = [(s.split("=")[0], int(s.split("=")[1])) for s in args.solvents]
xyz_files = [Path(x).resolve() for x in args.xyz]
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)

# --- solvent model: published ConSolv checkpoint, or our own trained one ----------
# A trained model carries a per-element energy shift (train.py's E0 refit); it cancels
# in BE because complex = rod + wheel in composition.
if args.params:
    run = Path(args.params).resolve()
    config = tomli.load(open(run / "config.toml", "rb"))
    assert config["model"]["type"] == "MACE_CONSOL", "binding needs a solvent model"
    params = consolv_model.load_params(run / "params.pkl")
else:
    config = tomli.load(open(UPSTREAM / "solvent/config.toml", "rb"))
    params = consolv_model.published_solvent_params(
        UPSTREAM / "vacuum/best_params_1.pkl", UPSTREAM / "solvent/full_init_params.pkl",
        UPSTREAM / "solvent/final_params.pkl")

csv = out / "binding.csv"  # extend an existing table; a rerun replaces matching rows
rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
for xyz in xyz_files:
    name = Path(xyz).stem
    atoms0 = read(xyz)
    n = len(atoms0)
    Z = atoms0.get_atomic_numbers()

    # rod / wheel = the two covalently disconnected components; wheel has no N
    rcov = covalent_radii[Z]
    graph = atoms0.get_all_distances() < 1.2 * (rcov[:, None] + rcov[None, :])
    ncomp, label = csgraph.connected_components(graph, directed=False)
    assert ncomp == 2, f"{name}: expected 2 components, got {ncomp}"
    wheel_label = [c for c in range(2) if 7 not in Z[label == c]]
    assert len(wheel_label) == 1, f"{name}: cannot tell wheel from rod"
    wheel_mask = label == wheel_label[0]
    rod_mask = ~wheel_mask
    print(f"{name}: {n} atoms, rod {rod_mask.sum()}, wheel {wheel_mask.sum()}", flush=True)

    # model for this system size (box + cutoff as in evaluate_molecule.py)
    R0 = atoms0.positions * 0.1
    box = jnp.eye(3) * 50.0
    displacement_fn, _ = space.periodic(jnp.diag(box))
    data = {"R": R0[None], "mask": onp.ones((1, n), bool), "species": Z[None],
            "solv_species": onp.full((1, n), solvents[0][1])}
    nbrs_init, (_, max_edges, avg_nn) = graphs.allocate_neighborlist(
        data, displacement_fn, box, config["model"]["r_cutoff"], mask_key="mask",
        format=partition.Sparse, fractional_coordinates=False, capacity_multiplier=2.0)
    _, energy_fn_template = consolv_model.build(config["model"], displacement_fn, max_edges, avg_nn)
    energy_fn = energy_fn_template(params)

    @jax.jit
    def energy(R, mask, solv):
        nbrs = nbrs_init.update(R, mask=mask)
        e = energy_fn(R, nbrs, species=jnp.asarray(Z), mask=mask,
                      solv_species=jnp.full(n, solv))
        return e, nbrs.did_buffer_overflow

    @jax.jit
    def energy_grad(R, solv):
        mask = jnp.ones(n, bool)
        (e, ovf), g = jax.value_and_grad(lambda r: energy(r, mask, solv), has_aux=True)(R)
        return e, g, ovf

    class ConSolvCalc(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, solv):
            super().__init__()
            self.solv = solv

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            e, g, ovf = energy_grad(jnp.asarray(self.atoms.positions * 0.1), self.solv)
            assert not ovf, "neighbour-list overflow"
            self.results["energy"] = float(e) / KJ_PER_EV
            self.results["forces"] = -onp.asarray(g) / KJ_PER_EV / 10.0

    for solv_name, solv in solvents:
        atoms = atoms0.copy()
        atoms.calc = ConSolvCalc(solv)
        opt = LBFGS(atoms, logfile=str(out / f"{name}_{solv_name}_opt.log"))
        converged = opt.run(fmax=args.fmax, steps=args.steps)
        write(out / f"{name}_{solv_name}_relaxed.xyz", atoms)

        R = atoms.positions * 0.1
        R_sep = R.copy()
        R_sep[wheel_mask] += [0.0, 5.0, 0.0]  # wheel 50 A away: same internal geometries
        E = {}
        for part, pos, mask in [("complex", R, onp.ones(n, bool)),
                                ("rod", R_sep, rod_mask), ("wheel", R_sep, wheel_mask)]:
            e, ovf = energy(jnp.asarray(pos), jnp.asarray(mask), solv)
            assert not ovf, "neighbour-list overflow"
            E[part] = float(e) * KCAL_PER_KJ
        be = E["complex"] - E["rod"] - E["wheel"]
        print(f"{name} {solv_name}: steps={opt.nsteps} converged={converged} BE={be:.2f} kcal/mol", flush=True)
        rows = [r for r in rows if (r["system"], r["solvent"]) != (name, solv_name)]
        rows.append({"system": name, "solvent": solv_name, "solv_species": solv,
                     "opt_steps": opt.nsteps, "converged": bool(converged),
                     "E_complex": E["complex"], "E_rod": E["rod"], "E_wheel": E["wheel"],
                     "BE_kcal": be})
        pd.DataFrame(rows).to_csv(csv, index=False, float_format="%.3f")
