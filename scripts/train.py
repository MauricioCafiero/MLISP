"""Force-match a ConSolv-style MACE on GFN2 energies + forces.

--model vacuum   the vacuum MACE prior (MACE), trained on gas-phase data
--model solvent  the solvent-conditional model (MACE_CONSOL), trained jointly on
                 GFN2/ALPB data in several solvents; each frame carries its ConSolv
                 solvent index (solv_species)

Uses chemtrain's ForceMatching trainer, as ConSolv's own vacuum training does.
Energies are referenced to GFN2 isolated-atom energies (E0s), the same way the
published models are referenced to MACE-OFF wB97M E0s: U = E - sum_i E0(Z_i).
Data sets are read from data/gfn2_<set>_{train,valid}.xyz.

    python scripts/train.py --model vacuum --sets gas --name vac_ft
    python scripts/train.py --model solvent --sets hexane=25 chloroform=11 acetone=7 dmf=53 water=1 --name solv_ft

--init published  start from the published checkpoint (vacuum or solvent)
--init scratch    fresh init, same architecture
Writes runs/<name>/{params.pkl, config.toml, metrics.csv, training.log}.
"""
import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]

import numpy as onp
import optax
import pandas as pd
import tomli
import tomli_w
from ase.io import read
from jax import random, tree_util
from jax_md import partition

import consolv_model
from chemtrain import trainers
from chemtrain.data import graphs
from chemtrain.learn import max_likelihood
from jax_md_mod import custom_space

KJ_PER_EV = 96.48533212
# GFN2-xTB isolated-atom energies (eV), from ~/python_mac/mace/code/finetune_mace.py (GFN2_E0S)
GFN2_E0S = {1: -10.707211, 6: -48.798080, 7: -70.908087, 8: -102.521807, 9: -125.698643,
            15: -64.604696, 16: -85.619453, 17: -121.975722, 35: -110.160925, 53: -102.848978}

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["vacuum", "solvent"], required=True)
p.add_argument("--sets", nargs="+", required=True, help="vacuum: gas; solvent: name=solv_species ...")
p.add_argument("--data-dir", default=str(ROOT / "data"))
p.add_argument("--init", choices=["published", "scratch"], default="published")
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--lr", type=float, default=2.5e-4)
p.add_argument("--gamma-U", type=float, default=1.0)
p.add_argument("--gamma-F", type=float, default=0.01)
p.add_argument("--clip", type=float, default=10.0, help="global grad-norm clip")
p.add_argument("--no-e0-refit", action="store_true",
               help="skip the per-element energy shift fitted to the initial model's residuals")
p.add_argument("--name", required=True)
args = p.parse_args()
sets = [(s.split("=")[0], int(s.split("=")[1]) if "=" in s else 0) for s in args.sets]
data_dir = Path(args.data_dir).resolve()
out = ROOT / "runs" / args.name
out.mkdir(parents=True, exist_ok=True)
os.chdir(out)  # chemtrain's trainer creates checkpoints/ in the cwd


def load(split, nmax):
    """extxyz (eV, A) -> chemtrain dataset (nm, kJ/mol), padded to nmax atoms.
    Sets are concatenated in order; relabelled sets share geometries frame for frame."""
    frames = [(f, solv) for name, solv in sets for f in read(data_dir / f"gfn2_{name}_{split}.xyz", ":")]
    n = len(frames)
    d = {"R": onp.zeros((n, nmax, 3)), "F": onp.zeros((n, nmax, 3)), "U": onp.zeros(n),
         "species": onp.zeros((n, nmax), int), "mask": onp.zeros((n, nmax), bool)}
    if args.model == "solvent":
        d["solv_species"] = onp.zeros((n, nmax), int)
    for i, (f, solv) in enumerate(frames):
        k = len(f)
        d["R"][i, :k] = f.positions * 0.1
        d["F"][i, :k] = f.arrays["REF_forces"] * KJ_PER_EV * 10.0
        d["U"][i] = (f.info["REF_energy"] - sum(GFN2_E0S[z] for z in f.numbers)) * KJ_PER_EV
        d["species"][i, :k] = f.numbers
        d["mask"][i, :k] = True
        if args.model == "solvent":
            d["solv_species"][i] = solv  # the model reads solv_species[0]; fill every slot
    d["U_weight"] = onp.ones(n)
    d["F_weight"] = nmax / d["mask"].sum(1)  # as ConSolv: per-frame force MSE over real atoms
    return d


nmax = max(len(f) for name, _ in sets for split in ("train", "valid")
           for f in read(data_dir / f"gfn2_{name}_{split}.xyz", ":"))
data = {"training": load("train", nmax), "validation": load("valid", nmax)}
print({k: v["U"].shape[0] for k, v in data.items()}, "frames, padded to", nmax, "atoms", flush=True)

config = tomli.load(open(ROOT / f"upstream/{args.model}/config.toml", "rb"))
displacement_fn, _ = custom_space.nonperiodic_general(fractional_coordinates=False)
both = tree_util.tree_map(lambda a, b: onp.concatenate([a, b]), data["training"], data["validation"])
nbrs_init, (_, max_edges, avg_nn) = graphs.allocate_neighborlist(
    both, displacement_fn, 0.0, config["model"]["r_cutoff"], mask_key="mask", format=partition.Sparse)
init_fn, energy_fn_template = consolv_model.build(config["model"], displacement_fn, max_edges, avg_nn)

d0 = data["training"]
nbrs0 = nbrs_init.update(d0["R"][0], mask=d0["mask"][0])
extra = {"solv_species": d0["solv_species"][0]} if args.model == "solvent" else {}
params = init_fn(random.PRNGKey(0), d0["R"][0], nbrs0, species=d0["species"][0], mask=d0["mask"][0], **extra)
if args.init == "published":
    up = ROOT / "upstream"
    pub = (consolv_model.load_params(up / "vacuum/best_params_1.pkl") if args.model == "vacuum" else
           consolv_model.published_solvent_params(up / "vacuum/best_params_1.pkl",
                                                  up / "solvent/full_init_params.pkl",
                                                  up / "solvent/final_params.pkl"))
    shapes = lambda t: {k: {kk: vv.shape for kk, vv in v.items()} for k, v in t.items()}
    assert shapes(pub) == shapes(params), "published params do not match the architecture"
    params = pub

steps = args.epochs * d0["U"].shape[0] // args.batch
optimizer = optax.chain(  # ConSolv's init_optimizer (Adam/Nesterov, decay, polynomial lr) + clip
    optax.clip_by_global_norm(args.clip),
    optax.scale_by_adam(b1=0.9, b2=0.99, eps=1e-8, eps_root=1e-4, nesterov=True),
    optax.transforms.add_decayed_weights(config["optimizer"]["weight_decay"]),
    optax.scale_by_learning_rate(optax.polynomial_schedule(args.lr, args.lr, 2, steps), flip_sign=True))

trainer = trainers.ForceMatching(
    params, optimizer, energy_fn_template, nbrs_init,
    batch_per_device=args.batch, batch_cache=10,
    gammas={"U": args.gamma_U, "F": args.gamma_F},
    weights_keys={"U": "U_weight", "F": "F_weight"},
    error_fns=max_likelihood.mse_loss, log_file=str(out / "training.log"))

# E0 refit: the starting model and the reference data use different isolated-atom
# references (published model: wB97M E0s; data: GFN2 E0s), a near-constant offset per
# element. Fit it to the starting model's residuals and remove it from the targets, so
# training spends no gradient on it. Complexes = rod + wheel make the composition matrix
# rank-deficient; lstsq's min-norm solution only absorbs per-element offsets. The shift is
# shared across solvents, so solvent-to-solvent differences are left for the model.
elements = sorted({int(z) for z in onp.unique(data["training"]["species"]) if z > 0})
counts = lambda d: onp.stack([(d["species"] == z).sum(1) for z in elements], 1).astype(float)
e0_shift = onp.zeros(len(elements))
if not args.no_e0_refit:
    resid = data["training"]["U"] - onp.asarray(trainer.predict(data["training"], params, batch_size=args.batch)["U"])
    e0_shift = onp.linalg.lstsq(counts(data["training"]), resid, rcond=None)[0]
    for split in data:
        data[split]["U"] = data[split]["U"] - counts(data[split]) @ e0_shift
    print("E0 shift (kJ/mol):", dict(zip(elements, e0_shift.round(2))), flush=True)

trainer.set_dataset(data["training"], stage="training")
trainer.set_dataset(data["validation"], stage="validation", include_all=True)


def metrics(p, split):
    """Energy RMSE (kcal/mol per frame and per atom), force RMSE (kcal/mol/A); for several
    sets also the RMSE of E(set) - E(first set) on the shared geometries (kcal/mol)."""
    d = data[split]
    pred = trainer.predict(d, p, batch_size=args.batch)
    n_at = d["mask"].sum(1)
    dU = (onp.asarray(pred["U"]) - d["U"]) / 4.184
    dF = (onp.asarray(pred["F"]) - d["F"])[d["mask"]] / 4.184 / 10.0
    m = {"U_rmse_kcal": float(onp.sqrt((dU ** 2).mean())),
         "U_rmse_kcal_per_atom": float(onp.sqrt(((dU / n_at) ** 2).mean())),
         "F_rmse_kcal_A": float(onp.sqrt((dF ** 2).mean()))}
    if len(sets) > 1:
        ref = d["U"].reshape(len(sets), -1) / 4.184
        prd = onp.asarray(pred["U"]).reshape(len(sets), -1) / 4.184
        for (name, _), r, q in list(zip(sets, ref - ref[0], prd - prd[0]))[1:]:
            m[f"dE_{name}_vs_{sets[0][0]}_rmse_kcal"] = float(onp.sqrt(((q - r) ** 2).mean()))
            m[f"dE_{name}_vs_{sets[0][0]}_ref_mean_kcal"] = float(r.mean())
    return m


before = metrics(params, "validation")
print("before:", before, flush=True)
trainer.train(args.epochs)
after = metrics(trainer.best_params, "validation")
print("after: ", after, flush=True)

pickle.dump(tree_util.tree_map(onp.asarray, trainer.best_params), open(out / "params.pkl", "wb"))
# energies from this model: E_ref = U_model + sum_i e0_shift[Z_i] + sum_i GFN2_E0S[Z_i]
tomli_w.dump({"model": config["model"], "args": vars(args), "nmax": nmax,
              "e0_shift_kJmol": {str(z): float(s) for z, s in zip(elements, e0_shift)}},
             open(out / "config.toml", "wb"))
pd.DataFrame([{"stage": "before", **before}, {"stage": "after", **after}]).to_csv(out / "metrics.csv", index=False)
print("wrote", out)
