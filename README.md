# consolv

Tools for running and training **solvent-conditional implicit-solvent machine-learning
potentials** in JAX, built on [ConSolv](https://github.com/tummfm/ConSolv) (Zhang &
Zavadlav, *ConSolv: Solvent-Conditional Machine Learning Implicit Solvent Potential*,
arXiv:2606.24983).

A ConSolv-style model is a MACE potential for a solute, conditioned on the solvent. A
solvent is an integer index into a table of 15 RDKit descriptors, covering 147 solvents.
The energy is

    U(R; solvent) = U_vacuum(R) + U_solvent(R; solvent)

which acts as a solvent-averaged potential for the solute. You get energies and forces
(`jax.grad`) for implicit-solvent optimisation, MD and binding energies at roughly
vacuum cost.

This repository:

- runs the **published ConSolv checkpoint** on stock, current-enough JAX/haiku;
- computes **binding energies** E(complex) − E(A) − E(B) of non-covalent or mechanically
  bonded complexes in any of the 147 solvents;
- **trains / fine-tunes** the vacuum MACE prior on your own reference data (e.g. GFN2-xTB)
  with chemtrain's force-matching trainer.

Training the solvent-conditional head on your own data is in progress (see `DEV.md`).

## Install

Python 3.11, CPU JAX:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
uv pip install --python .venv/bin/python --no-deps -e ./third_party/chemtrain -e ./third_party/chemutils
.venv/bin/python scripts/fetch_upstream.py     # published ConSolv checkpoint + descriptor table -> upstream/
```

The pins in `requirements.lock.txt` matter: the JAX ecosystem has to be from the jax-0.4.30
era. See `DEV.md` for why.

## Usage

**Solvent index from SMILES:**

```sh
.venv/bin/python scripts/solvent_index.py water=O 'acetone=CC(C)=O' hexane=CCCCCC
# water 1, acetone 7, hexane 25
```

**Binding energies.** The complex is relaxed in each solvent; the two components are
single points at their geometries in the relaxed complex:

```sh
.venv/bin/python scripts/binding_energy.py complex1.xyz complex2.xyz \
    --solvents water=1 acetone=7 hexane=25
# -> results/binding.csv, results/<name>_<solvent>_relaxed.xyz
```

Input: an xyz file (Å) with exactly two covalently disconnected components. Currently the
component without nitrogen is taken as the "wheel" (written for rotaxanes; generalise
`binding_energy.py` for other complexes).

**Reference data with GFN2-xTB.** Uses `gfn2_data.py` from a separate
[MACE fine-tuning repo](https://github.com/MauricioCafiero/MACE_UseAndTrain) and its env
(tblite):

```sh
<mace-env>/bin/python scripts/make_gfn2_data.py complex1.xyz complex2.xyz   # gas phase
<mace-env>/bin/python scripts/make_gfn2_data.py ... --solvent water          # GFN2/ALPB
# same geometries in a solvent (no new sampling):
<mace-env>/bin/python scripts/make_gfn2_data.py --relabel data/gfn2_gas_train.xyz data/gfn2_gas_valid.xyz --solvent water
```

**Fine-tune / train.** Data sets are read from `data/gfn2_<set>_{train,valid}.xyz`:

```sh
# vacuum prior on gas-phase data
.venv/bin/python scripts/train.py --model vacuum --sets gas --name my_vac
# solvent-conditional model on ALPB data in several solvents (name=solvent index)
.venv/bin/python scripts/train.py --model solvent --sets hexane=25 acetone=7 water=1 --name my_solv
# --init published (default) or scratch -> runs/<name>/{params.pkl, config.toml, metrics.csv}
```

`train.py` refits the per-element energy reference to the starting model before training.
This is needed whenever the reference method's isolated-atom energies differ from the
checkpoint's (see `DEV.md`).

## Layout

```
scripts/
  consolv_model.py     model construction + checkpoint loading
  haiku_patch.py       compatibility patch the published model needs
  binding_energy.py    implicit-solvent binding energies
  solvent_index.py     SMILES -> solvent index
  make_gfn2_data.py    GFN2-xTB reference data (gas / ALPB)
  train.py             force-matching fine-tune / training (vacuum or solvent model)
  fetch_upstream.py    downloads the published ConSolv files
third_party/           vendored chemtrain + chemutils (Apache-2.0, see NOTICE)
upstream/              published ConSolv files (downloaded, not committed)
```

## Credits and licences

This project's code and docs are MIT-licensed (`LICENSE`). Vendored code in `third_party/`
keeps its own Apache-2.0 licence.

- **ConSolv** (method, published checkpoint, solvent descriptor table): Linying Zhang and
  Julija Zavadlav, TU Munich. Please cite their paper if you use this. ConSolv's top-level
  files carry no licence, so they are not redistributed here: `scripts/fetch_upstream.py`
  downloads them from a pinned commit.
- **chemtrain** and **chemutils** (Multiscale Modeling of Fluid Materials, TU Munich) are
  vendored under `third_party/` under the Apache License 2.0. Changes are listed in
  `NOTICE`.

```bibtex
@article{zhang2026consolv,
  title   = {ConSolv: Solvent-Conditional Machine Learning Implicit Solvent Potential},
  author  = {Zhang, Linying and Zavadlav, Julija},
  journal = {arXiv preprint arXiv:2606.24983},
  year    = {2026}
}
```
