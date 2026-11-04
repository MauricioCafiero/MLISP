import subprocess
from pathlib import Path

import mdtraj
import numpy as onp
import pandas as pd

from jax_md_mod import io


DEFAULT_SOURCE_CSV = (
    Path(__file__).resolve().parents[4]
    / "examples/spice/postprocess/benchmark/cosmo_sfe_error_by_heavy_atoms_upto19_plot_data.csv"
)


def download_cosmo_exp_solvatum_upto19(
        root="./_data",
        scale_R=1.0,
        scale_U=4.184,
        fractional=True,
        source_csv=None,
        regenerate=False,
        **kwargs):
    """Preprocess the 491 COSMO/experimental matched pairs as a test set.

    The source CSV is the previously filtered xlsx-vs-Solv@TUM subset with
    solute heavy atom count up to 19. The source experimental values are read
    in kcal/mol and scaled to kJ/mol in ``dF`` by the default ``scale_U=4.184``.
    COSMO-RS values are retained in metadata.
    """
    dataset, info = download_and_process(
        root=root,
        source_csv=source_csv,
        regenerate=regenerate,
    )
    dataset = scale_dataset(dataset, scale_R, scale_U=scale_U, fractional=fractional)

    return {
        "training": None,
        "validation": None,
        "testing": dataset,
    }, info


def download_and_process(root, source_csv=None, regenerate=False):
    data_dir = Path(root)
    data_dir.mkdir(exist_ok=True, parents=True)
    data_dir = data_dir / "cosmo_exp_solvatum_upto19"
    data_dir.mkdir(exist_ok=True, parents=True)

    if (data_dir / "database.npz").exists() and not regenerate:
        dataset = dict(onp.load(data_dir / "database.npz"))
        info = get_dataset_info(data_dir)
        return dataset, info

    df = download_raw(data_dir, source_csv=source_csv)

    conf_dir = data_dir / "confs"
    conf_dir.mkdir(exist_ok=True)

    raw_data = {
        "R": [],
        "dF": [],
        "species": [],
        "mass": [],
        "bonds": [],
        "solv_species": [],
        "task": [],
        "mol_id": [],
    }
    info = []

    allowed_atomic_numbers = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
    unique_solutes = df[["solute_smiles", "solute_index"]].drop_duplicates()

    for _, row in unique_solutes.iterrows():
        mol_id = row["solute_index"]
        solute_smiles = row["solute_smiles"]
        pdb_path = conf_dir / f"{mol_id}.pdb"

        if not pdb_path.exists():
            generate_conformer(solute_smiles, pdb_path)

        _, pos, mass, species = io.load_box(str(pdb_path))
        if onp.all(pos == 0.0):
            print(f"Regenerate all-zero conformation {mol_id} for {solute_smiles}.")
            generate_conformer(solute_smiles, pdb_path)
            _, pos, mass, species = io.load_box(str(pdb_path))

        unallowed_atomic = set(species.tolist()) - allowed_atomic_numbers
        if unallowed_atomic:
            print(
                f"The molecule {solute_smiles} contains unallowed atoms "
                f"{unallowed_atomic}. Skipping."
            )
            continue

        top = mdtraj.load_topology(str(pdb_path))
        bondlist = onp.zeros((2, 2 * top.n_bonds), dtype=int)
        for idx, (b1, b2) in enumerate(top.bonds):
            bondlist[0, 2 * idx] = b1.index
            bondlist[1, 2 * idx] = b2.index
            bondlist[0, 2 * idx + 1] = b2.index
            bondlist[1, 2 * idx + 1] = b1.index

        if onp.all(pos == 0.0):
            print(f"Skip conformation {mol_id} with all-zero positions.")
            continue

        solute_data = df[df["solute_smiles"] == solute_smiles]

        for _, data_point in solute_data.iterrows():
            raw_data["R"].append(pos)
            raw_data["species"].append(species)
            raw_data["mass"].append(mass)
            raw_data["dF"].append(float(data_point["solvation_free_energy"]))
            raw_data["bonds"].append(bondlist)
            raw_data["solv_species"].append(int(data_point["solvent_index"]))
            raw_data["task"].append(int(data_point["solvent_index"]))
            raw_data["mol_id"].append(data_point["solute_index"])
            info.append(data_point)

    if not raw_data["R"]:
        raise RuntimeError("No valid datapoints were preprocessed.")

    write_smiles_csv(data_dir / "smiles.csv", info)

    max_atoms = max(len(species) for species in raw_data["species"])
    max_bonds = max(bonds.shape[1] for bonds in raw_data["bonds"])
    n_samples = len(raw_data["R"])

    data = {
        "id": onp.arange(n_samples, dtype=int),
        "R": onp.zeros((n_samples, max_atoms, 3), dtype=float),
        "dF": onp.zeros((n_samples,), dtype=float),
        "species": onp.zeros((n_samples, max_atoms), dtype=int),
        "mask": onp.zeros((n_samples, max_atoms), dtype=bool),
        "mass": onp.zeros((n_samples, max_atoms), dtype=float),
        "bonds": onp.full((n_samples, 2, max_bonds), max_atoms, dtype=int),
        "solv_species": onp.zeros((n_samples, max_atoms), dtype=int),
        "task": onp.zeros(n_samples, dtype=int),
        "mol_id": onp.empty(n_samples, dtype="U8"),
    }

    for idx, entry in enumerate(zip(*raw_data.values())):
        pos, free_eng, species, mass, bonds, solv_species, task_num, mol_id = entry

        data["R"][idx, :mass.size] = pos
        data["dF"][idx] = free_eng
        data["mask"][idx, :mass.size] = True
        data["species"][idx, :mass.size] = species
        data["mass"][idx, :mass.size] = mass
        data["bonds"][idx, :, :bonds.shape[1]] = bonds
        data["solv_species"][idx, :mass.size] = solv_species
        data["task"][idx] = task_num
        data["mol_id"][idx] = mol_id

    onp.savez(data_dir / "database.npz", **data)

    if n_samples != 491:
        print(f"Warning: preprocessed {n_samples} datapoints instead of 491.")

    return data, get_dataset_info(data_dir)


def download_raw(data_dir, source_csv=None):
    source_csv = Path(source_csv) if source_csv is not None else DEFAULT_SOURCE_CSV
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    output_path = data_dir / "cosmo_exp_solvatum_upto19.csv"
    df = pd.read_csv(source_csv)
    required_columns = {
        "smiles_solvent",
        "smiles_solute",
        "selected_solvent_ids",
        "selected_solvent_names",
        "experimental_kcal_mol",
        "cosmo_rs_kcal_mol",
        "error_cosmo_minus_experiment_kcal_mol",
        "absolute_error_kcal_mol",
        "solute_heavy_atoms",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")

    df = df[df["solute_heavy_atoms"].astype(int) <= 19].copy()
    if df["selected_solvent_ids"].astype(str).str.contains(",", regex=False).any():
        raise ValueError("Found ambiguous selected_solvent_ids with comma-separated IDs.")
    if len(df) != 491:
        raise ValueError(f"Expected 491 source datapoints, found {len(df)}.")

    solute_identity = df["inchi_solute"].fillna(df["smiles_solute"]).astype(str)
    representative_smiles = representative_solute_smiles(df, solute_identity)
    solute_indices = smiles_index_mapping(solute_identity.tolist())
    processed = pd.DataFrame({
        "data_origin": "xlsx_cosmo_exp_upto19",
        "solvent_name": df["selected_solvent_names"].astype(str),
        "solvent_index": df["selected_solvent_ids"].astype(int),
        "solvent_smiles": df["smiles_solvent"].astype(str),
        "solute_index": [solute_indices[key] for key in solute_identity],
        "solute_smiles": [representative_smiles[key] for key in solute_identity],
        "source_solute_smiles": df["smiles_solute"].astype(str),
        "solvation_free_energy": df["experimental_kcal_mol"].astype(float),
        "cosmo_rs_free_energy": df["cosmo_rs_kcal_mol"].astype(float),
        "cosmo_rs_error": df["error_cosmo_minus_experiment_kcal_mol"].astype(float),
        "cosmo_rs_abs_error": df["absolute_error_kcal_mol"].astype(float),
        "solute_heavy_atoms": df["solute_heavy_atoms"].astype(int),
        "solvent_heavy_atoms": df["solvent_heavy_atoms"].astype(int),
        "inchi_solute": df["inchi_solute"].astype(str),
        "inchi_solvent": df["inchi_solvent"].astype(str),
        "source_datapoint_index": df["datapoint_index"].astype(int),
    })
    processed.to_csv(output_path, index=False)
    print(f"491-point source data saved to {output_path}")
    return processed


def generate_conformer(solute_smiles, pdb_path):
    command = ["obabel", "-ismi", "-O", str(pdb_path), "--gen3d", "slowest"]
    print(f"Run command {' '.join(command)} for {solute_smiles}")
    subprocess.run(command, input=solute_smiles, text=True, check=True)


def representative_solute_smiles(df, solute_identity):
    representatives = {}
    grouped = df.assign(_solute_identity=solute_identity).groupby(
        "_solute_identity", sort=False
    )
    for identity, group in grouped:
        smiles = group["smiles_solute"].astype(str).tolist()
        if "[H][H]" in smiles:
            representatives[identity] = "[H][H]"
        else:
            representatives[identity] = smiles[0]
    return representatives


def smiles_index_mapping(identity_list):
    unique_identities = {}
    index = 1

    for identity in identity_list:
        if identity not in unique_identities:
            unique_identities[identity] = f"mol_{index:04d}"
            index += 1
    return unique_identities


def write_smiles_csv(path, info):
    rows = []
    for idx, data_point in enumerate(info):
        rows.append({
            "id": idx,
            "mol_id": data_point["solute_index"],
            "solute_smiles": data_point["solute_smiles"],
            "source_solute_smiles": data_point.get("source_solute_smiles", data_point["solute_smiles"]),
            "solvent_id": int(data_point["solvent_index"]),
            "solvent_smiles": data_point["solvent_smiles"],
            "data_origin": data_point["data_origin"],
            "solvent_name": data_point["solvent_name"],
            "solvation_free_energy": float(data_point["solvation_free_energy"]),
            "cosmo_rs_free_energy": float(data_point["cosmo_rs_free_energy"]),
            "cosmo_rs_error": float(data_point["cosmo_rs_error"]),
            "cosmo_rs_abs_error": float(data_point["cosmo_rs_abs_error"]),
            "solute_heavy_atoms": int(data_point["solute_heavy_atoms"]),
            "solvent_heavy_atoms": int(data_point["solvent_heavy_atoms"]),
            "inchi_solute": data_point["inchi_solute"],
            "inchi_solvent": data_point["inchi_solvent"],
            "source_datapoint_index": int(data_point["source_datapoint_index"]),
        })

    pd.DataFrame(rows).to_csv(path, sep=";", index=False)


def get_dataset_info(data_dir):
    df = pd.read_csv(data_dir / "smiles.csv", delimiter=";", dtype=str, comment=None)

    info_dict = {}
    for _, row in df.iterrows():
        info_dict[int(row["id"])] = {
            "key": row["mol_id"],
            "smile": row["solute_smiles"],
            "solv_id": row["solvent_id"],
            "solv_smi": row["solvent_smiles"],
            "properties": {
                "data_origin": row["data_origin"],
                "solvent_name": row["solvent_name"],
                "source_solute_smiles": row.get("source_solute_smiles", row["solute_smiles"]),
                "solvation_free_energy": float(row["solvation_free_energy"]),
                "cosmo_rs_free_energy": float(row["cosmo_rs_free_energy"]),
                "cosmo_rs_error": float(row["cosmo_rs_error"]),
                "cosmo_rs_abs_error": float(row["cosmo_rs_abs_error"]),
                "solute_heavy_atoms": int(row["solute_heavy_atoms"]),
                "solvent_heavy_atoms": int(row["solvent_heavy_atoms"]),
                "inchi_solute": row["inchi_solute"],
                "inchi_solvent": row["inchi_solvent"],
                "source_datapoint_index": int(row["source_datapoint_index"]),
            },
        }
    return info_dict


def scale_dataset(dataset, scale_R, scale_U, fractional=True):
    box = 10 * (dataset["R"].max(axis=(1, 2)) - dataset["R"].min(axis=(1, 2)))

    if fractional:
        dataset["R"] = dataset["R"] / box[:, onp.newaxis, onp.newaxis]
    else:
        dataset["R"] = dataset["R"] * scale_R

    dataset["dF"] = dataset["dF"] * scale_U
    dataset["box"] = (
        scale_R
        * box[:, onp.newaxis, onp.newaxis]
        * onp.tile(onp.eye(3), (box.size, 1, 1))
    )

    return dataset


if __name__ == "__main__":
    dataset, info = download_cosmo_exp_solvatum_upto19(
        root="/ds/project/linying/Datasets",
        fractional=False,
        regenerate=True,
    )
    print(f"Testing datapoints: {dataset['testing']['id'].size}")
