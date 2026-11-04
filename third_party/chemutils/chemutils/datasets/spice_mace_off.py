"""Downloads and prepares the SPICE dataset."""

from urllib import request
from pathlib import Path

import h5py

import re
import jax
import numpy as onp

from chemtrain.data import preprocessing
from chemutils.datasets import utils
import tarfile
from ase.io import read

spice_versions: dict[str, str] = {
    'mace_off_test': "https://www.repository.cam.ac.uk/bitstreams/b185b5ab-91cf-489a-9302-63bfac42824a/download",
    'mace_off_train': "https://www.repository.cam.ac.uk/bitstreams/cb8351dd-f09c-413f-921c-67a702a7f0c5/download",
}

# Atomic energies
E0s={35: -70045.28385080204, 6: -1030.5671648271828, 17: -12522.649269035726, 9: -2715.318528602957, 1: -13.571964772646918, 53: -8102.524593409054, 7: -1486.3750255780376, 8: -2043.933693071156, 15: -9287.407133426237, 16: -10834.4844708122}


def download_spice(root="./_data",
                   scale_R=0.0529177,
                   scale_U=2625.5,
                   scale_e=11.7871,
                   fractional=True,
                   max_samples=None,
                   version='mace_off',
                   subsets=None,
                   **kwargs):
    """Download complete SPICE dataset.

    Args:
        root: Download directory of SPICE dataset.
        scale_R: Scaling factor for atomic positions (default=5.2917721e-2 #Bohr -> nm).
        scale_U: Scaling factor for potential energies (default=2625.5 #Hartee -> kJ/mol).
        fractional: Whether to scale positions by simulation box (default=true).
        url: Url for dataset download (default refers to SPICE v2.0.1).
        subsets: List of regex to select subsets of the dataset
        max_samples: Maximum number of samples to select (proportional to subset size)

    Returns:
        The complete SPICE dataset.
    """

    with jax.default_device(jax.devices("cpu")[0]):
        data_dir = download_source(spice_versions["mace_off_train"], root=root)
        data_dir = download_source(spice_versions["mace_off_test"], root=root)

        train_dataset = load_and_padd_samples(data_dir, "train_large_neut_no_bad_clean.xyz", "train")
        test_dataset = load_and_padd_samples(data_dir, "test_large_neut_all.xyz", "test")
        
        train_dataset, train_subsets = select_subsets(data_dir, train_dataset, subsets, "train")
        test_dataset, test_subsets = select_subsets(data_dir, test_dataset, subsets, "test")
        
        # Apply dataset filter
        test_dataset = filter_dataset(test_dataset)

        # Apply max_samples filter if specified
        if max_samples is not None:
            train_dataset = partial_subset(train_dataset, max_samples)
            test_dataset = partial_subset(test_dataset, max_samples)
        
        train_dataset = scale_dataset(train_dataset, scale_R, scale_U, scale_e, fractional)
        test_dataset = scale_dataset(test_dataset, scale_R, scale_U, scale_e, fractional)
        
        # Combine into final dataset structure
        dataset = {
            "training": train_dataset,
            "validation": test_dataset,
            "testing": test_dataset
        }

    info = {
        "subsets": train_subsets,
        "version": spice_versions,
        "scaling": {
            "R": scale_R,
            "U": scale_U,
            "fractional": fractional
        },
    }

    return dataset, info


def download_source(url: str, root: str="./_data"):
    """Downloads and unpacks the SPICE dataset. 'https://zenodo.org/records/8222043' gives an overview on the available
    versions.

    Urls of available versions of the SPICE dataset:
    v1.1.2: 'https://zenodo.org/records/7338495/files/SPICE-1.1.2.hdf5?download=1'
    v1.1.3: 'https://zenodo.org/records/7606550/files/SPICE-1.1.3.hdf5?download=1'
    v1.1.4: 'https://zenodo.org/records/8222043/files/SPICE-1.1.4.hdf5?download=1'
    v2.0.0: 'https://zenodo.org/records/10835749/files/SPICE-2.0.0.hdf5?download=1'
    v2.0.1: 'https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5?download=1'

    Args:
        url: Url for dataset download.
        root: Download directory of SPICE dataset.

    Returns:
        Path of the download directory.
    """
    data_dir = Path(root)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    if not (data_dir / "SPICE_mace_off_all").exists():
        (data_dir / "SPICE_mace_off_all").mkdir(exist_ok=True, parents=True)
        print(f"Download SPICE dataset from {url}")
        archive_path = data_dir / "SPICE_mace_off_all/SPICE_mace_off.tar.gz"
        request.urlretrieve(url, archive_path, utils.show_progress)
        print("Unpacking SPICE_mace_off dataset...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=data_dir / "SPICE_mace_off_all")
        archive_path.unlink()

    return data_dir / "SPICE_mace_off_all"


def load_and_padd_samples(data_dir, filename, split_name):
    """Loads and padds the atom data."""

    # Do not process more than once
    npz_filename = f"SPICE_{split_name}.npz"
    if (data_dir / npz_filename).exists():
        return dict(onp.load(data_dir / npz_filename))

    # The original dataset is in extended xyz format
    file_path = data_dir / filename
    print(f"Reading data from {file_path}")
    frames = list(read(file_path, index=':'))
    print(f"Loaded {len(frames)} frames from xyz file")
    
    # First pass: discover subsets and max atoms
    # Check if subsets file already exists
    subsets_file = data_dir / f"subsets_train.dat"
    if subsets_file.exists():
        print(f"Loading existing subsets from {subsets_file}")
        with open(subsets_file, "r") as file:
            subsets = [line.partition(" ")[2].strip("\n") for line in file.readlines()]
        subsets.append("Tripeptide")
    else:
        subsets = []
        for frame in frames:
            current_subset = frame.info.get("config_type")
            if current_subset and current_subset not in subsets:
                print(f"Discovered new subset: {current_subset}")
                subsets.append(current_subset)

    max_atoms = 0
    n_samples = len(frames)
    for frame in frames:
        n_atoms = len(frame.arrays["numbers"])
        if n_atoms > max_atoms:
            max_atoms = n_atoms

    print(f"Found {n_samples} samples with a maximum of {max_atoms} atoms.")

    # Reserve memory for the complete padded dataset
    dataset = {
        "id": onp.zeros((n_samples,), dtype=int),
        "R": onp.zeros((n_samples, max_atoms, 3)),
        "F": onp.zeros((n_samples, max_atoms, 3)),
        "U": onp.zeros((n_samples,)),
        "subset": onp.zeros((n_samples,), dtype=int),
        "species": onp.zeros((n_samples, max_atoms), dtype=int),
        "mask": onp.zeros((n_samples, max_atoms), dtype=bool),
    }

    # Second pass: populate dataset
    for idx, frame in enumerate(frames):
        n_atoms = len(frame.arrays["numbers"])
        dataset["id"][idx] = idx
        subset_str = frame.info.get("config_type")
        if subset_str and subset_str in subsets:
            dataset["subset"][idx] = subsets.index(subset_str)
        
        dataset["mask"][idx, :n_atoms] = True
        dataset["species"][idx, :n_atoms] = frame.arrays["numbers"]
        dataset["R"][idx, :n_atoms, :] = frame.arrays["positions"]
        
        # if "MACE_forces" in frame.arrays:
        dataset["F"][idx, :n_atoms, :] = frame.get_forces()
        
        # if "MACE_energy" in frame.info:
        dataset["U"][idx] = frame.get_potential_energy() - sum(E0s[Z] for Z in frame.arrays["numbers"])

    # save dataset
    onp.savez(data_dir / npz_filename, **dataset)
    with open(data_dir / f"subsets_{split_name}.dat", "w") as file:
        file.writelines([f"{str(idx).rjust(3, '0')} {subset}\n"
                         for idx, subset in enumerate(subsets)])

    return dataset


def process_dataset(dataset):
    """Creates weights for masked loss."""
    print(dataset)
    for split in dataset.keys():
        # Weight the potential by the number of particles. The per-particle
        # potential should have equal weight, so the error for larger systems
        # should contribute more. On the other hand, since we compute the
        # MSE for the forces, we have to correct for systems with masked
        # out particles.

        n_particles = onp.sum(dataset[split]['mask'], axis=1)
        max_particles = dataset[split]['mask'].shape[1]

        dataset[split]['U_weight'] = onp.ones_like(n_particles)
        dataset[split]['F_weight'] = max_particles / n_particles
        print(f"energies in KJ/mol: {dataset[split]['U'].min()} to {dataset[split]['U'].max()}")

    return dataset


def select_subsets(data_dir, dataset, subsets, split_name):
    """Selects the subsets from the dataset."""
    if subsets is None:
        # Return all data without filtering
        with open(data_dir / f"subsets_{split_name}.dat", "r") as file:
            subset_info = {
                int(line.partition(" ")[0]): line.partition(" ")[2].strip("\n")
                for line in file.readlines()
            }
        return dataset, subset_info

    with open(data_dir / f"subsets_{split_name}.dat", "r") as file:
        selection: dict[int, str] = {
            int(line.partition(" ")[0]): line.partition(" ")[2]
            for line in file.readlines()
            if any([re.search(s, line) for s in subsets])
        }

    boolean_mask = onp.isin(dataset["subset"], list(selection.keys()))
    dataset = {
        key: arr[boolean_mask] for key, arr in dataset.items()
    }

    return dataset, {idx: sel.strip("\n") for idx, sel in selection.items()}


def partial_subset(dataset, max_samples):
    """Selects a maximum number of samples proportional to each subset size."""
    
    total_samples = dataset["subset"].size
    unique_subsets = onp.unique(dataset["subset"])
    keys = dataset.keys()
    
    selected_indices = []
    
    for subset_id in unique_subsets:
        # Get indices for this subset
        subset_mask = dataset["subset"] == subset_id
        subset_indices = onp.where(subset_mask)[0]
        subset_size = subset_indices.size
        
        # Calculate max samples for this subset proportional to total
        max_subset = int(subset_size / total_samples * max_samples)
        max_subset = min(max_subset, subset_size)  # Don't exceed actual subset size
        
        # Select the first max_subset samples from this subset
        selected_indices.extend(subset_indices[:max_subset].tolist())
    
    # Convert to array and filter dataset
    selected_indices = onp.array(selected_indices)
    filtered_dataset = {
        key: arr[selected_indices] for key, arr in dataset.items()
    }
    
    return filtered_dataset


def filter_dataset(dataset):
    # According to MACE-OFF, filter out samples with abs(F) > 15 eV/Angstrom and sum(F) > 0.1
    mask = (onp.abs(dataset["F"]) <= 15).all(axis=(1, 2)) & (onp.abs(dataset["F"].sum(axis=(1, 2))) <= 0.1)
    filtered_dataset = {key: arr[mask] for key, arr in dataset.items()}
    return filtered_dataset 


def scale_dataset(dataset, scale_R, scale_U, scale_e, fractional=True):
    """Scales the dataset from Hartee to kJ/mol and Bohr to nm."""
    box = 10 * (dataset["R"].max() - dataset["R"].min())

    print(f"Original positions: {dataset['R']}")
    print(f"Original positions: {dataset['R'].min()} to {dataset['R'].max()}")

    if fractional:
        dataset['R'] = dataset['R'] / box
    else:
        dataset['R'] = dataset['R'] * scale_R

    print(f"Scale dataset by {scale_R} for R and {scale_U} for U.")

    scale_F = scale_U / scale_R
    dataset['box'] = scale_R * onp.tile(box * onp.eye(3), (dataset['R'].shape[0], 1, 1))
    print(f"original energies: {dataset['U'].min()} to {dataset['U'].max()}")
    dataset['U'] *= scale_U
    dataset['F'] *= scale_F

    return dataset


if __name__ == "__main__":
    # U: eV --> kJ/mol (1 eV = 96.485 kJ/mol)
    # R: Angstrom --> nm (1 Angstrom = 0.1 nm)
    dataset, info = download_spice(root="/home/linying/Datasets", scale_U=96.485, scale_R=0.1, max_samples=10000, fractional=False)

    dataset = process_dataset(dataset)
    
    for key in dataset.keys():
        print(f"Split has species in range {dataset[key]['species'].min()} to {dataset[key]['species'].max()}")
        print(f"Unique values are {onp.unique(dataset[key]['species'])}")