# Dev notes

Working notes: what was needed to get ConSolv running, what has been verified, and what
is open. User-facing docs are in `README.md`.

## Status (2026-11-05)

| piece | state |
|---|---|
| Published ConSolv checkpoint runs (CPU, jax 0.4.30) | done; regression-checked after vendoring |
| Binding energies in any solvent (`binding_energy.py`) | done; fragment evaluation verified (below) |
| GFN2 reference data (`make_gfn2_data.py`) | gas sampling + ALPB relabel both run |
| (a) Vacuum MACE force-matching (`train.py --model vacuum`) | mini test passed (below) |
| Upstream clone | removed; nothing depends on it |
| Fresh-venv install from README | tested: reproduces regression exactly |
| (b) Solvent model trained on GFN2/ALPB in 5 solvents (`train.py --model solvent`) | mini test passed (below) |
| 7-rotaxane GFN2 test (`ROTX7.md`): data, 4 models on Modal, held-out BE | done; see `ROTX7.md` |
| Vacuum + delta solvent scheme (`train.py --delta-from`, `binding_energy.py --vacuum-params`) | done; best solvent response |
| MPS / jax-metal | not tried |

## Where the fixes to upstream live

Nothing in upstream ConSolv was edited. Every fix is in this repo:

| problem with upstream as cloned | fix | where |
|---|---|---|
| `environment.yml` does not resolve to a working env | exact pins | `requirements.lock.txt`; reasons below |
| Model code calls a haiku feature no release has | runtime patch | `scripts/haiku_patch.py` (gotcha 1) |
| Checkpoint keys nested for the authors' patched haiku | un-nest on load | `consolv_model.load_params` (gotcha 2) |
| Model building/loading lives in upstream `examples/SFE/train/train_utils.py`, which has no licence and pulls in mlflow, plotting and every model type | own minimal reimplementation (MACE + MACE_CONSOL only) | `scripts/consolv_model.py`: `build`, `merge`, `published_solvent_params` (gotcha 3) |
| Checkpoints/configs/descriptor table have no licence | download at a pinned commit, never committed | `scripts/fetch_upstream.py` → `upstream/` |
| chemtrain/chemutils are git subdirs of an unlicensed repo, with docs/examples (19 MB) | vendor package sources only, Apache-2.0 | `third_party/`, `NOTICE` |
| chemutils reads the descriptor table from upstream's `dataset/` | path → `upstream/` | `third_party/chemutils/chemutils/models/mace/test_block.py` (`NOTICE`) |
| chemutils `pyproject.toml` says `README.md`, file is `Readme.md` | rename | `third_party/chemutils/` (`NOTICE`) |
| In-place fragment masking silently ignored | evaluate fragments separated by 50 Å | `scripts/binding_energy.py` (gotcha 5) |
| Upstream scripts hard-code cluster paths and CUDA devices | not used; our scripts take paths as arguments | `scripts/*` |

Regression check after moving off the clone (vendored packages + `upstream/` +
`consolv_model.py`): rot2/acetone at the input geometry gives E_complex −12169.066,
E_rod −6727.690, E_wheel −5425.503 kcal/mol and BE −15.873. That is identical to the
original clone-based run.

## Environment pins and why

Upstream `environment.yml` is conda + `jax[cuda12]==0.4.31` and pins none of the JAX
ecosystem. It does not resolve to a working set.

| package | version | why |
|---|---|---|
| Python | 3.11 (uv) | 3.14 (Homebrew default) too new for jax 0.4.30 |
| jax / jaxlib | 0.4.30 | chemtrain requires `<=0.4.30`; env.yml's 0.4.31 contradicts it |
| jax-md | git `7fdbba1` (2024-07-31) | PyPI 0.2.8 uses removed `jax.random.KeyArray`; main needs newer jax (`jax.extend.backend.get_backend`) |
| dm-haiku | 0.0.12 | 0.0.17 needs `jax.core.take_current_trace` |
| chex / optax | 0.1.86 / 0.2.3 | jax-0.4.30 era |
| numpy / scipy | 1.26.4 / 1.12.0 | as upstream (chemtrain: scipy < 1.13) |
| mlflow | 3.8.1 | was imported by upstream `train_utils.py`; no longer needed by our code, can probably drop |

## Gotchas in the published model

1. **Patched haiku.** ConSolv calls `hk.MultiHeadAttention(...)(q, k, v, return_attn_weights=...)`,
   which no released haiku has. `scripts/haiku_patch.py` adds it: haiku 0.0.12's `__call__`
   verbatim, returning `(out, weights)` on request. It asserts the haiku version.
2. **Param-key nesting.** The published pickles store attention params under
   `.../attention/~/multi_head_attention/{query,key,value,linear}`; stock haiku expects
   `.../attention/{...}`. `consolv_model.load_params` un-nests them.
3. **Solvent checkpoint = delta.** The published solvent model is
   `final_params + merge(full_init_params, vacuum_params renamed mace→mace_consol)`.
   `merge` takes the vacuum array only where key and shape match (this reproduces upstream's
   `simple_recursive_merge` exactly). See `consolv_model.published_solvent_params`.
4. **Solvent index** = `descriptor.csv` row + 1 (row 0 of the model's table is all zeros,
   i.e. vacuum). The model reads only `solv_species[0]`, so every atom must carry the same
   index.
5. **Neighbour-list masking is lazy.** jax-md's `nbrs.update(R, mask=...)` only rebuilds once
   atoms have moved. Masking a fragment in place at an unmoved geometry is silently ignored,
   so the fragments sum exactly to the complex and BE ≡ 0. `binding_energy.py` therefore
   evaluates fragments with the wheel translated 50 Å away (receptive field 2 × 5 Å).
   Verified: BE(separated) = 0.000, and fragment energies are identical whether taken from
   the bound or the separated geometry.
6. **Units.** chemtrain uses nm and kJ/mol. The published vacuum model's energies are
   E − Σ E0(Z) with MACE-OFF wB97M E0s (SPICE converted from Bohr/Hartree).
   `train.py` references GFN2 data to GFN2 isolated-atom E0s the same way.
7. `chemutils` reads the descriptor table at import-time path
   `<repo>/upstream/descriptor.csv` (changed from `dataset/`; see NOTICE).

## Published checkpoint: training-solvent coverage

`choose_task` in the published solvent config (also hard-coded in upstream
`resolv_parallel_add.py`) lists the 66 solvents the solvent head was trained on. It includes
acetone (7), hexane (25), pentane (40), chloroform (11), DCM (17), DMF (53), formamide (88),
toluene (45). It does **not** include water (1) or DMSO (104). Upstream `task_filter` has a
water-specific subsampling branch, so water was used in some runs, just not in this
checkpoint's final list (unconfirmed against the paper). Treat water/DMSO predictions from
this checkpoint as extrapolation.

Training solutes look small (an upstream loader is named `..._upto19`, possibly ≤ 19 heavy
atoms). Systems of 100+ atoms are out of distribution for the published model.

## POC findings (published checkpoint)

Systems: rot1 (144 atoms; rod 88 + 24-crown-8 wheel 56) and rot2 (114; rod 58 + wheel 56),
starting from UMA-relaxed wheel-on-one-side structures. BE = E(complex) − E(rod) − E(wheel)
in kcal/mol. The complex is relaxed (LBFGS, fmax 0.02 eV/Å, all converged); rod and wheel are
single points at their geometries in the relaxed complex.

| solvent | ε | index | in training list | rot1 BE | rot2 BE |
|---|---|---|---|---|---|
| formamide | 109 | 88 | yes | −21.93 | −23.80 |
| water | 78 | 1 | **no** | −25.24 | −25.57 |
| DMSO | 47 | 104 | **no** | −24.27 | −24.50 |
| DMF | 37 | 53 | yes | −23.13 | −23.42 |
| acetone | 21 | 7 | yes | −23.70 | −24.02 |
| DCM | 8.9 | 17 | yes | −26.22 | −25.70 |
| chloroform | 4.8 | 11 | yes | −25.91 | −25.49 |
| n-hexane | 1.88 | 25 | yes | −28.54 | −27.81 |
| n-pentane | 1.84 | 40 | yes | −28.38 | −27.78 |

Across the training solvents, BE strengthens almost monotonically as polarity falls
(formamide → hexane: 6.6 kcal/mol for rot1, 4.0 for rot2; only sub-0.4 kcal/mol inversions).
Water and DMSO, which are not in the training list, break the pattern. Implicit-solvent DFT
experience (user, protein–ligand) says binding should be *weakest* in the least polar solvent.
The published solvent head does not reproduce that trend for these systems, which motivates
training our own.

## (a) Vacuum fine-tune mini test

Data: `make_gfn2_data.py` on rot1/rot2. 5 molecules (rot1 complex 144, rot2 complex 114,
rot1 rod 88, rot2 rod 58, wheel 56 atoms), GFN2 relax + 300 K Langevin MD (400 × 0.5 fs,
every 20 steps): 100 train / 10 valid frames. About 6 min; the first run used 5 parallel
processes (don't, see Conventions). Training: published vacuum checkpoint, 10 epochs,
batch 4, lr 2.5e-4, γ_U = 1, γ_F = 0.01, CPU, about 0.47 min/epoch.

Validation RMSE:

| run | E (kcal/mol/frame) | E (kcal/mol/atom) | F (kcal/mol/Å) |
|---|---|---|---|
| published, as is | 3221 | 32.8 | 4.75 |
| `gfn2_gas_ft`: fine-tuned, no E0 refit | 163 | 2.6 | **19.1** |
| published after E0 refit (no training) | 2.76 | 0.028 | 4.75 |
| `gfn2_gas_ft2`: E0 refit + grad clip 10 | 2.65 | 0.028 | 4.16 |

- **E0 refit is required** when the reference level differs from the starting model's.
  The published model sits on wB97M E0s and the data on GFN2 E0s: about 33 kcal/mol/atom of
  pure offset. Without the refit, energy gradients swamp the forces and wreck the weights.
  `train.py` fits a per-element shift to the starting model's residuals (lstsq,
  min-norm; complex = rod + wheel makes it rank-deficient), removes it from the targets and
  stores it in `runs/<name>/config.toml`. Fitted shifts: H −165, C −137, N −17, O −55,
  F −116 kJ/mol.
- The "Gradient norm" in chemtrain's training.log is the **squared** norm
  (`chemtrain.util.tree_norm` returns Σ x·x despite its docstring). The logged 1e10–1e13 are
  real norms of ~1e5–3e6. They fluctuate but do not grow. `train.py --grad-report` at the
  published vacuum start: norm 6.1e5, 92% of it from the single global
  `atomic_energy_layer/~/scale_shift_layer` scale. Adam normalises per parameter, so that
  does not swamp the rest, and clipping before Adam barely changes step sizes. Loss terms
  are balanced (F 3.3e4 × γ_F 0.01 vs U 2.4e2). No loss normalisation needed.
  `train.py` writes the true (square-rooted) per-step norms to `runs/<name>/grad_norms.csv`.
- chemtrain's trainer creates `checkpoints/` in the cwd; `train.py` chdirs to
  `runs/<name>/`.

## (b) Solvent-model mini test

Data: the same 110 gas frames relabelled (`make_gfn2_data.py --relabel`) with GFN2/ALPB in
hexane (ConSolv index 25), chloroform (11), acetone (7), DMF (53) and water (1): 500 train /
50 valid frames on identical geometries. About 10 min, serial. Training:
`train.py --model solvent`, published ConSolv solvent checkpoint, E0 refit (shared across
solvents), 5 epochs, batch 4, lr 2.5e-4. About 2.6 min total on CPU. The whole network was
trained on total ALPB energies and forces (nothing frozen).

| validation | published (after E0 refit) | after 5 epochs |
|---|---|---|
| E RMSE (kcal/mol/frame) | 7.30 | 3.37 |
| F RMSE (kcal/mol/Å) | 4.93 | 4.17 |

E(solvent) − E(hexane) on identical geometries (kcal/mol):

| solvent | ALPB reference mean | RMSE before | RMSE after |
|---|---|---|---|
| chloroform | −3.19 | 3.05 | 1.52 |
| acetone | +0.21 | 5.10 | 2.69 |
| DMF | +0.34 | 3.28 | 2.61 |
| water | +19.78 | 18.71 | 5.38 |

The per-frame `solv_species` reaches the solvent head through chemtrain's dynamic kwargs:
every non-target key of the dataset is passed to the energy function. The model learns the
solvent differences. The published checkpoint has essentially none of the large ALPB
water-vs-hexane shift, and water is not in its training list. Caveats: tiny data;
validation frames come from the same MD runs as training frames; whole-network fine-tune.

## BE with the ALPB-trained model vs GFN2/ALPB

`binding_energy.py --params runs/alpb5_solv_ft` (complex relaxed with the trained model) vs
the published model. Reference: `gfn2_binding.py`, GFN2/ALPB single points of complex, rod and
wheel on each model's own relaxed geometries. BE in kcal/mol.

| system | solvent | published | GFN2/ALPB (pub. geoms) | ALPB-trained | GFN2/ALPB (trained geoms) |
|---|---|---|---|---|---|
| rot1 | hexane | −28.54 | −18.55 | −24.12 | −18.26 |
| rot1 | chloroform | −25.91 | −16.16 | −22.04 | −15.91 |
| rot1 | acetone | −23.70 | −12.96 | −19.41 | −12.58 |
| rot1 | DMF | −23.13 | −12.61 | −18.91 | −12.25 |
| rot1 | water | −25.24 | −22.36 | −20.06 | −22.16 |
| rot2 | hexane | −27.81 | −20.82 | −23.36 | −21.38 |
| rot2 | chloroform | −25.49 | −18.29 | −22.11 | −18.60 |
| rot2 | acetone | −24.02 | −16.74 | −19.55 | −15.59 |
| rot2 | DMF | −23.42 | −16.39 | −19.07 | −15.18 |
| rot2 | water | −25.57 | −24.19 | −20.56 | −24.36 |

MAE vs GFN2/ALPB: rot1 8.78 → 5.52, rot2 5.97 → 3.43. Spread across solvents: GFN2/ALPB 9–10,
both models ~4–5.

- Training removes a uniform 3–4 kcal/mol of over-binding. About 6 kcal/mol of over-binding
  remains in the organic solvents, which sits in the vacuum part (interaction energy still
  mostly from the wB97M-trained prior after 5 epochs).
- Ordering among the organic solvents (hexane > chloroform > acetone ≈ DMF) agrees between
  both models and GFN2/ALPB.
- GFN2/ALPB gives the strongest binding in water (nonpolar/cavity term). The trained model
  turns up in water by ~1 kcal/mol vs ALPB's ~10: right direction, far too small.
- Reference BE moves < ~1 kcal/mol between the two models' geometries, so this compares
  energies, not structures.

## 7-rotaxane GFN2 dataset (`data/rotx7/`)

Inputs: `inputs/current_structures/` (gitignored; threaded starting geometries):
rot1, rot2htpuma, rot3, dethread1–4 (104–144 atoms; wheels 24C8 or DB24C8). XYZ order is rod
first, then wheel. `make_gfn2_data.py --fragments` takes the rod as the covalent component
holding atom 1 (verified against the SMILES formulas for all 7). Driver: `scripts/run_rotx7.sh`
(resumable, detached).

Per complex: GFN2 gas relax + 1200 × 0.5 fs Langevin MD at 300 K, sampled every 30 steps →
42 frames (relaxed minimum + 41 MD). Every frame is cut into complex, rod and wheel; the
fragments get gas single points. Everything is then relabelled with GFN2/ALPB in hexane,
chloroform, acetone, DMF, water and DMSO (ConSolv indices 25, 11, 7, 53, 1, 104).
Validation = dethread4 held out entirely.

| set | train frames | valid frames |
|---|---|---|
| gas + each of 6 solvents | 756 (6 systems × 42 × 3 parts) | 126 (dethread4) |

Timing, serial on the laptop (2026-11-05): MD + gas labels ~41 min for all 7 complexes
(~5–7 min each); each solvent relabel of 882 frames ~2.5–3 min (~0.2 s per single point).
The mini-test timings above were inflated: gas sampling ran 5 processes at once.

## Next steps (not done)

- Improve the vacuum model's host–guest interaction: it over-binds held-out dethread4 by
  6.4 kcal/mol in the gas phase and dominates the remaining BE error (`ROTX7.md`).
- Production: label with DFT at the published level, ωB97M-D3(BJ)/def2-TZVPPD (SPICE) and train on rotaxane
  frames plus a small subset of the original training data to limit forgetting. See `ROTX7.md`
  "Notes for production training".
- Use the delta scheme (a) for production (vacuum model + solvent model on E_solv − E_gas).
- Optional: forgetting test on unseen solvents (relabel held-out frames in DCM, toluene, ...).

## Training on Modal

`scripts/modal_train.py` runs `train.py` on an L4 (falls back to A10G; L4s have been scarce),
app `mlisp-train`, results on volume `mlisp-runs` (`modal volume get mlisp-runs <name> runs/`).
Launch detached and give each launch its own local log:
`modal run --detach scripts/modal_train.py --which vacuum|solvent|delta|embed`.

- Cost on an L4 (rotx7, batch 4): vacuum 756 frames ~0.2 min/epoch; solvent 4536 frames
  ~1.25–1.45 min/epoch (first epoch +1 min JIT). The vacuum run was 10 min, the solvent runs
  25–50 min.
- Gotchas: the lock file's `dataclasses==0.8` (py3.6 backport) is skipped in the image; the
  script re-imports inside the container, so the lock file is read only when `modal.is_local()`;
  a stopped-then-relaunched app keeps retrying unless stopped with `modal app stop -y <id>`.
- `train.py` writes `config.toml` with `None`-valued args removed (a crash there cost the
  config of `rotx7_delta` and `rotx7_embed`; both were rebuilt from their logs, noted inside).

## Local-only tools (not in requirements.lock.txt)

Installed into the repo `.venv` for the laptop side only: `modal` (launches
`scripts/modal_train.py`; the Modal image is built from the lock file), `markdown` and `xhtml2pdf`
(`scripts/md_to_pdf.py`), `pypdfium2` (rendering PDF pages to check them).

## Conventions

- Run heavy local compute serially (`make_gfn2_data.py --workers` defaults to 1).
- Research inputs/outputs (`inputs/ data/ results/ runs/`) are gitignored; the
  structures may be unpublished.
