"""GFN2-xTB training data for the rotaxanes, via the mace repo's gfn2_data.py.

Molecules: each complex in inputs/*.xyz, its rod, and the wheel (taken once:
24-crown-8 is the same in rot1 and rot2). Each is GFN2-relaxed, then sampled by
Langevin MD; molecules run one at a time (--workers N to parallelise small
jobs only -- the laptop does not like many concurrent GFN2 runs). Writes data/gfn2_<solv>_<label>.xyz
(extxyz, eV / Angstrom, REF_energy / REF_forces) and merged
data/gfn2_<solv>_{train,valid}.xyz.

Run with the mace repo's env (tblite lives there; this script imports no torch):
    ~/python_mac/mace/.venv/bin/python scripts/make_gfn2_data.py inputs/rot1.xyz inputs/rot2.xyz
    ... --solvent water        # GFN2/ALPB instead of gas phase

--fragments: MD on the complexes only; every sampled complex frame is cut into rod
(the covalent component holding atom 1) and wheel, which get gas single points. Per
system -> <out>/gfn2_gas_<sys>.xyz (info part = complex/rod/wheel; skipped if it exists,
so reruns resume). Merged by system: --valid-systems go to valid, the rest to train:
    ... scripts/make_gfn2_data.py --fragments --out data/rotx7 --md-steps 1200 \
        --sample-every 30 --valid-systems dethread4 inputs/current_structures/*.xyz
<sys> is the xyz stem up to the first '_'.

--relabel: single-point existing extxyz frames in an ALPB solvent instead (no new
sampling), so every solvent sees the same geometries:
    ... scripts/make_gfn2_data.py --relabel data/gfn2_gas_train.xyz data/gfn2_gas_valid.xyz --solvent water
    -> data/gfn2_water_train.xyz, data/gfn2_water_valid.xyz
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse.csgraph as csgraph
from ase.data import covalent_radii
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "python_mac/mace/code"))
import gfn2_data  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("xyz", nargs="+")
p.add_argument("--solvent", default="gas", help="gas, or an ALPB solvent name")
p.add_argument("--md-steps", type=int, default=400)
p.add_argument("--sample-every", type=int, default=20)
p.add_argument("--temperature", type=float, default=300.0)
p.add_argument("--workers", type=int, default=1)
p.add_argument("--relabel", action="store_true", help="xyz args are extxyz frames to re-label")
p.add_argument("--fragments", action="store_true", help="MD on complexes; label complex, rod and wheel frames")
p.add_argument("--valid-systems", type=lambda s: s.split(","), default=[], help="--fragments: comma-separated systems held out as validation")
p.add_argument("--out", default=str(ROOT / "data"))
args = p.parse_args()
out = Path(args.out)
out.mkdir(exist_ok=True)

if args.relabel:
    assert args.solvent != "gas", "--relabel is for ALPB solvents"
    for path in args.xyz:
        dest = out / Path(path).name.replace("gas", args.solvent)
        if dest.exists():
            print(f"{dest} exists, skipped", flush=True)
            continue
        frames = read(path, ":")
        for f in frames:
            f.calc = None
            res = gfn2_data.gfn2_singlepoint(f, solvation=("alpb", args.solvent))
            f.calc = None
            f.info.update(REF_energy=res["energy"], solvent=args.solvent)
            f.info.pop("energy", None)
            f.arrays["REF_forces"] = res["forces"]
            f.arrays.pop("forces", None)
        write(dest.with_suffix(".tmp.xyz"), frames, format="extxyz")
        dest.with_suffix(".tmp.xyz").rename(dest)
        print(f"{path} -> {dest} ({len(frames)} frames)", flush=True)
    sys.exit()


def split(atoms):
    """(rod, wheel) = the two covalent components; the wheel has no N."""
    r = covalent_radii[atoms.numbers]
    graph = atoms.get_all_distances() < 1.2 * (r[:, None] + r[None, :])
    ncomp, label = csgraph.connected_components(graph, directed=False)
    assert ncomp == 2, f"expected 2 components, got {ncomp}"
    wheel = [c for c in range(2) if 7 not in atoms.numbers[label == c]][0]
    return atoms[label != wheel], atoms[label == wheel]


if args.fragments:
    assert args.solvent == "gas", "--fragments labels gas phase; use --relabel for ALPB"
    systems = []
    for xyz in args.xyz:
        sysname = Path(xyz).stem.split("_")[0]
        dest = out / f"gfn2_gas_{sysname}.xyz"
        systems.append((sysname, dest))
        if dest.exists():
            print(f"{dest} exists, skipped", flush=True)
            continue
        cplx = read(xyz)
        r = covalent_radii[cplx.numbers]
        ncomp, label = csgraph.connected_components(cplx.get_all_distances() < 1.2 * (r[:, None] + r[None, :]))
        nrod = int((label == label[0]).sum())
        assert ncomp == 2 and (label[:nrod] == label[0]).all(), f"{xyz}: not rod atoms 1..{nrod} + wheel"
        cplx.info["label"] = sysname
        md = out / f"gfn2_gas_{sysname}_md.xyz"
        if not md.exists():
            gfn2_data.generate_dataset([cplx], outfile=str(md), temperatures_K=(args.temperature,),
                                       md_steps=args.md_steps, sample_every=args.sample_every,
                                       do_normal_modes=False)
        frames = []
        for f in read(md, ":"):
            f.info["part"] = "complex"
            frames.append(f)
            for part, sub in (("rod", f[:nrod]), ("wheel", f[nrod:])):
                sub.calc = None
                res = gfn2_data.gfn2_singlepoint(sub)
                sub.info.pop("energy", None)
                sub.arrays.pop("forces", None)
                sub.info.update(REF_energy=res["energy"], part=part)
                sub.arrays["REF_forces"] = res["forces"]
                frames.append(sub)
        write(dest.with_suffix(".tmp.xyz"), frames, format="extxyz")
        dest.with_suffix(".tmp.xyz").rename(dest)
        print(f"{sysname}: {len(cplx)} atoms (rod {nrod}), {len(frames) // 3} frames x 3 parts", flush=True)
    for part, keep in (("train", lambda n: n not in args.valid_systems), ("valid", lambda n: n in args.valid_systems)):
        frames = [f for n, dest in systems if keep(n) for f in read(dest, ":")]
        write(out / f"gfn2_gas_{part}.xyz", frames, format="extxyz")
        print(f"{part}: {len(frames)} frames -> gfn2_gas_{part}.xyz", flush=True)
    sys.exit()

molecules = []
for i, xyz in enumerate(args.xyz):
    name = Path(xyz).stem
    cplx = read(xyz)
    rod, wheel = split(cplx)
    molecules += [(f"{name}_complex", cplx), (f"{name}_rod", rod)]
    if i == 0:
        molecules.append(("wheel", wheel))


def run(item):
    label, atoms = item
    atoms.info["label"] = label
    stem = out / f"gfn2_{args.solvent}_{label}"
    calc_kw = {} if args.solvent == "gas" else {"solvation": ("alpb", args.solvent)}
    res = gfn2_data.generate_dataset(
        [atoms], outfile=f"{stem}.xyz", train_file=f"{stem}_train.xyz", valid_file=f"{stem}_valid.xyz",
        temperatures_K=(args.temperature,), md_steps=args.md_steps, sample_every=args.sample_every,
        do_normal_modes=False, **calc_kw)
    return label, len(atoms), res["n_train"], res["n_valid"]


if __name__ == "__main__":
    if args.workers > 1:
        pool = Pool(args.workers)
        results = pool.imap_unordered(run, molecules)
    else:
        results = map(run, molecules)
    for label, n, ntr, nva in results:
        print(f"{label}: {n} atoms, {ntr} train / {nva} valid frames", flush=True)
    for part in ("train", "valid"):
        frames = [f for lab, _ in molecules for f in read(out / f"gfn2_{args.solvent}_{lab}_{part}.xyz", ":")]
        write(out / f"gfn2_{args.solvent}_{part}.xyz", frames)
        print(f"{part}: {len(frames)} frames -> gfn2_{args.solvent}_{part}.xyz")
