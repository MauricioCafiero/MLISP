"""Model construction + parameter loading for ConSolv-style MACE potentials.

Thin layer over the Apache-2.0 chemutils MACE builders (mace_neighborlist_pp
for the vacuum model, mace_consol_neighborlist_pp for the solvent-conditional
one). Units follow chemtrain: nm, kJ/mol.
"""
import pickle

import jax.numpy as jnp
from jax import tree_util

import haiku_patch  # noqa: F401  (adds return_attn_weights to hk.MultiHeadAttention)
from chemutils.models.mace import mace_consol_neighborlist_pp, mace_neighborlist_pp

N_SPECIES = 55       # atomic-number embedding size used by the published models
N_SOL_SPECIES = 148  # 147 descriptor rows + the zero (vacuum) row


def build(model_cfg, displacement_fn, max_edges, avg_num_neighbors):
    """(init_fn, energy_fn_template) for model_cfg["type"] in {"MACE", "MACE_CONSOL"}.

    energy_fn_template(params)(R, nbrs, species=..., mask=..., [solv_species=...]) -> U
    """
    kw = dict(max_edges=max_edges, output_irreps="1x0e", per_particle=False,
              avg_num_neighbors=avg_num_neighbors, mode="energy", positive_species=True,
              **model_cfg["model_kwargs"])
    r_cut = model_cfg["r_cutoff"]
    if model_cfg["type"] == "MACE":
        init_fn, gnn = mace_neighborlist_pp(displacement_fn, r_cut, N_SPECIES, **kw)
    elif model_cfg["type"] == "MACE_CONSOL":
        init_fn, gnn = mace_consol_neighborlist_pp(displacement_fn, r_cut, N_SPECIES, N_SOL_SPECIES, **kw)
    else:
        raise ValueError(model_cfg["type"])

    def energy_fn_template(params):
        def energy_fn(pos, neighbor, mode=None, system=None, duplicate=None, **kwargs):
            kwargs.setdefault("mask", jnp.ones(pos.shape[0], bool))
            return gnn(params, pos, neighbor, **kwargs)
        return energy_fn

    return init_fn, energy_fn_template


def load_params(path):
    """Pickle -> jnp pytree. The authors' patched haiku nests MHA one level deeper
    than stock haiku (.../attention/~/multi_head_attention/query); un-nest it."""
    p = tree_util.tree_map(jnp.asarray, pickle.load(open(path, "rb")))
    return {k.replace("/attention/~/multi_head_attention/", "/attention/"): v for k, v in p.items()}


def merge(new, old):
    """Keys/structure from `new`; take `old`'s array wherever the key exists and the
    shape matches (keys only in `old` are dropped)."""
    if isinstance(new, dict):
        return {k: merge(v, old[k]) if isinstance(old, dict) and k in old else v for k, v in new.items()}
    return old if hasattr(old, "shape") and old.shape == new.shape else new


def published_solvent_params(vacuum_pkl, full_init_pkl, final_pkl):
    """Published ConSolv solvent model = final (a delta) + vacuum merged into the full init."""
    vac = load_params(vacuum_pkl)
    vac = {k.replace("mace", "mace_consol") if "mace" in k else k: v for k, v in vac.items()}
    vac = {k.replace("linear_node_embedding_layer", "linear_node_embedding_block"): v for k, v in vac.items()}
    full_vac = merge(load_params(full_init_pkl), vac)
    return tree_util.tree_map(jnp.add, load_params(final_pkl), full_vac)
