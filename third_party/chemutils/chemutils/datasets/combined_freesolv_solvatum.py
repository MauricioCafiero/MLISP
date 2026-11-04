import os
from pathlib import Path
from urllib import request, parse
import zipfile
import json
import subprocess
from collections import defaultdict
import requests

import mdtraj
import numpy as onp
import pandas as pd

from jax_md_mod import io
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from chemtrain.data import preprocessing
from chemutils.datasets import utils

import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_SOLVATUM_ROOT = _REPOSITORY_ROOT / "dataset" / "solvatum"
if str(_SOLVATUM_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOLVATUM_ROOT))

from solvatum.ui import Database


_urls = {
    "0.51": "https://escholarship.org/content/qt6sd403pz/supp/FreeSolv-0.51.zip"
}
import jax.tree_util as tree_util

def task_filter1(dataset, num):
    '''Training dataset filtering by tast training datapoints.'''
    leaves, _ = tree_util.tree_flatten(dataset)
    task_ids = leaves[-1]
    # Create a dictionary to hold indices by task
    task_indices = defaultdict(list)
    for idx, task_id in enumerate(list(task_ids)):
        hashable_id = task_id.item() if hasattr(task_id, 'item') else str(task_id)
        task_indices[hashable_id].append(idx)

    task_indices = {task_id: task_indices[task_id] for task_id in task_indices.keys() if len(task_indices[task_id])>num}
    indices_list = [idx for task_ids in task_indices.values() for idx in task_ids]
    return task_indices

def download_combined(root="./_data",
                      scale_R=1.0,
                      scale_U=4.184,
                      fractional=True,
                      version='0.51',
                      subsets=None,
                      split="random",
                      regenerate=False,
                      **kwargs):
    dataset, info = download_and_process(_urls[version], version, root=root, subsets=subsets, regenerate=regenerate)
    dataset = scale_dataset(dataset, scale_R, scale_U, fractional)
    if split == "random":
        training, validation, testing = preprocessing.train_val_test_split(
            dataset, **kwargs, shuffle=True, shuffle_seed=11)
    elif split == "stratified":
        training, validation, testing = preprocessing.stratified_train_val_test_split_for_tasks(
            dataset, **kwargs, shuffle=True, shuffle_seed=123)
    elif split == "stratified_mol":
        training, validation, testing = preprocessing.stratified_train_val_test_split_for_mols(
            dataset, **kwargs, shuffle=True, shuffle_seed=123)
    
    else:
        training = dataset
        validation = None
        testing = None

    dataset = {
        "training": training,
        "validation": validation,
        "testing": testing
    }

    return dataset, info

def download_and_process(url: str, version: str, root: str="./_data", subsets=None, regenerate: bool=False):
    """Downloads and prepares the FreeSolv and solvatum database for use with the ReSolv method.

    This function generates initial conformations for all SMILES strings using
    the command-line tool OpenBabel (must be installed).

    """
    data_dir = Path(root)
    data_dir.mkdir(exist_ok=True, parents=True)

    data_dir = data_dir / "conditional_solvent"
    if (data_dir / "database.npz").exists() and regenerate is False:
        dataset = dict(onp.load(data_dir / "database.npz"))
        info = get_dataset_info(data_dir)
        return dataset, info
    
    df = download_raw(url, version, data_dir)

    # Create initial conformations using openbabel
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
        "mol_id": []
    }

    info = []

    # Process each unique solute
    unique_solutes = df[['solute_smiles', 'solute_index']].drop_duplicates()
    # Define allowed atomic numbers
    # TODO: double check the atom type in the SPICE dataset
    allowed_atomic_numbers = {1, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 35, 53}
    for _, row in unique_solutes.iterrows():
        mol_id = row['solute_index']
        solute_smiles = row['solute_smiles']
        
        # Generate 3D conformer if not exists
        if not (conf_dir / f"{mol_id}.pdb").exists():
            command = f'echo "{solute_smiles}" | obabel -ismi -O {conf_dir}/{mol_id}.pdb --gen3d slowest'
            print(f"Run command {command}")
            subprocess.run(command, check=True, shell=True)
        
        # Load molecule data
        _, pos, mass, species = io.load_box(f"{conf_dir}/{mol_id}.pdb")
        
        # Check for allowed atomic numbers
        unallowed_atomic = set(species.tolist()) - allowed_atomic_numbers
        if unallowed_atomic:
            print(f'The molecule {solute_smiles} contains unallowed atoms {unallowed_atomic}. Skipping.')
            continue
            
        top = mdtraj.load_topology(f"{conf_dir}/{mol_id}.pdb")
        
        # Create bondlist
        bondlist = onp.zeros((2, 2 * top.n_bonds), dtype=int)
        for idx, (b1, b2) in enumerate(top.bonds):
            bondlist[0, idx] = b1.index
            bondlist[1, idx] = b2.index
            # Make undirected
            bondlist[0, 2 * idx] = b2.index
            bondlist[1, 2 * idx] = b1.index
        
        # Skip invalid conformations
        if onp.all(pos == 0.0):
            print(f"Skip conformation {mol_id} with all-zero positions.")
            continue
        
        # Get all solvation data for this solute
        solute_data = df[df['solute_smiles'] == solute_smiles]
        
        for _, data_point in solute_data.iterrows():
            raw_data["R"].append(pos)
            raw_data["species"].append(species)
            raw_data["mass"].append(mass)
            raw_data["dF"].append(float(data_point['solvation_free_energy']))
            raw_data["bonds"].append(bondlist)
            raw_data["solv_species"].append(data_point['solvent_index'])
            raw_data["task"].append(data_point['solvent_index'])
            raw_data["mol_id"].append(data_point['solute_index'])
            info.append(data_point)

    # Save SMILES mapping with all data point information
    with open(data_dir / "smiles.csv", "w") as f:
        # Write header
        header = ["id", "mol_id", "solute_smiles", "solvent_id", "solvent_smiles", 
                "data_origin", "solvent_name", "solvation_free_energy"]
        f.write(";".join(header) + "\n")
        
        # Write data points
        for idx, data_point in enumerate(info):
            row = [
                str(idx),
                str(data_point['solute_index']),
                data_point['solute_smiles'],
                str(data_point['solvent_index']),
                data_point['solvent_smiles'],
                data_point['data_origin'],
                data_point['solvent_name'],
                str(data_point['solvation_free_energy']),
            ]
            f.write(";".join(row) + "\n")
    
    # Process raw data with padding
    max_atoms = max([len(species) for species in raw_data["species"]])
    max_bonds = max([bonds.shape[1] for bonds in raw_data["bonds"]])
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
        "mol_id": onp.empty(n_samples, dtype='U8')
    }
    
    # Fill the padded arrays
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
    
    # Save processed data
    onp.savez(data_dir / "database.npz", **data)
    
    return data, get_dataset_info(data_dir)

def download_raw(url: str, version: str, data_dir):
    # save solute_info_dict and solvent_info_dict
    output_path = data_dir / "combined_dataset.csv"
    if output_path.exists():
        df = pd.read_csv(output_path)
        print(f"Combined dataset loaded from {output_path}")
        return df

    ## import freesolv database
    if not (data_dir / "source.zip").exists():
        data_dir.mkdir()
        print(f"Download FREE_SOLV dataset from {url}")
        request.urlretrieve(url, data_dir / "source.zip", utils.show_progress)

        with zipfile.ZipFile(data_dir / "source.zip", 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    with open(data_dir / f"FreeSolv-{version}/database.json", "r") as f:
        database_freesolv = json.load(f)

    ## import solvatum database
    database_solvatum = Database()
    database_solvatum.energy_unit='kcal/mol'
    
    solute_smiles_list = []
    for key, value in database_freesolv.items():
        solute_smiles_list.append(value["smiles"])
    for key in database_solvatum.solutes:
        name = database_solvatum.solutes[key]
        value = database_solvatum.get_molecule_properties(name)
        solute_smiles_list.append(value["SMILES"])
    
    solute_info_dict = smiles_index_mapping(solute_smiles_list)

    solvent_info_dict = {}
    solvent_name_list = []
    solvent_name_list.append('water')
    for key in database_solvatum.solvents:
        solvent_name_list.append(key)

    for i, name in enumerate(solvent_name_list):
        solvent_info_dict[name] = {}
        # smiles
        try: 
            smiles = database_solvatum.get_molecule_properties(name)['SMILES']
        except:
            url = 'http://cactus.nci.nih.gov/chemical/structure/' + parse.quote(name) + '/smiles'
            smiles = request.urlopen(url).read().decode('utf8')
        solvent_info_dict[name]['SMILES'] = smiles
        # index
        solvent_info_dict[name]['index'] = i+1 # start from 1, vacuum for 0
        # solvent type
        solvent_info_dict[name]['solvent_type'] = classify_molecule(smiles)

    # combine two datasets
    df = generate_combined_dataset(database_freesolv, database_solvatum, solute_info_dict, solvent_info_dict)

    df.to_csv(output_path, index=False)
    print(f"Combined dataset saved to {output_path}")
    print(f"Total entries: {len(df)}")
    return df

def generate_combined_dataset(database_freesolv, database_solvatum, solute_info_dict, solvent_info_dict):
    """
    Generate combined dataset from FreeSolv and Solvatum databases
    """
    
    # Initialize list to store all data points
    data_points = []
    # Process FreeSolv database (water only)
    for key, value in database_freesolv.items():
        solute_smiles = value["smiles"]
        solvent_props = database_solvatum.get_molecule_properties('water')
        data_point = {
            'data_origin': 'freesolv',
            'solvent_name': 'water',
            'solvent_index': solvent_info_dict['water']['index'],
            'solvent_smiles': solvent_info_dict['water']['SMILES'],
            'solute_index': solute_info_dict[solute_smiles]['solute_index'],
            'solute_smiles': solute_smiles,
            'solvation_free_energy': value.get('expt', None),
        }
        data_points.append(data_point)

    missing_data = []
    for key in database_solvatum.solutes:
        solute_name = database_solvatum.solutes[key]
        solute_props = database_solvatum.get_molecule_properties(solute_name)
        solute_smiles = solute_props["SMILES"]
        # Get solvation data for this solute
        pro, solvation_data = database_solvatum.filtering(solute=solute_name)

        for log in solvation_data:
            # if log[0] in solvent_info_dict:
            log[0] = log[0].upper()
            try:
                solvent_props = database_solvatum.get_molecule_properties(log[0])
            except:
                missing_data.append(log[0]) if log[0] not in missing_data else None
                solute_props = {}
            
            data_point = {
                'data_origin': 'solvatum',
                'solvent_name': log[0],
                'solvent_index': solvent_info_dict[log[0]]['index'],
                'solvent_smiles': solvent_info_dict[log[0]]['SMILES'],
                'solute_index': solute_info_dict[solute_smiles]['solute_index'],
                'solute_smiles': solute_smiles,
                'solvation_free_energy': log[-1],
            }
            data_points.append(data_point)    

    print(f"missing data: {missing_data}, total number: {len(missing_data)}")       
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data_points)
    return df

def classify_molecule(smiles):
    """Classify molecule based on functional groups and other properties"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ['invalid_smiles']
    
    classifications = []

    functional_groups = {
        # Highest Priority Organic Functional Groups (tend to be most reactive/polar)
        'carboxylic_acid': '[CH,O]=O[OH]',  # e.g., Acetic Acid
        'ester': '[#6][CX3](=O)[OX2H0]',    # e.g., Ethyl Acetate
        'amide': '[NX3][CX3](=[OX1])[#6]',    # e.g., Dimethylformamide (DMF)
        'phosphate': '[P](=O)([O])',       # e.g., Tributyl Phosphate (Note: pattern matches core P=O with attached O's)
        'nitrile': '[NX1]#[CX2]',           # e.g., Acetonitrile
        'aldehyde': '[CX3H1](=O)[#6]',      # e.g., Acetaldehyde (less common bulk solvent)
        'ketone': '[#6][CX3](=O)[#6]',      # e.g., Acetone
        'sulfone': '[S;X4](=O)=O',         # e.g., Sulfolane (Sulfur double-bonded to two oxygens, bonded to two other atoms)
        'sulfoxide': '[S;X3](=O)',         # e.g., Dimethyl Sulfoxide (DMSO) (Sulfur double-bonded to one oxygen, bonded to two other atoms)
        'alcohol': '[OH]',                  # e.g., Ethanol, Methanol
        'amine': '[NH2]',                   # e.g., Butylamine (less common as general solvent)

        # Lower Priority Organic Functional Groups / Substituents
        'sulfide': '[#16X2H0]',             # e.g., Diethyl Sulfide (Sulfur single-bonded to two carbons)
        'ether': '[OR]',                    # e.g., Diethyl Ether, THF
        'halogen': '[F,Cl,Br,I]',           # e.g., Dichloromethane, Chlorobenzene

        # Structural Classes / Other Solvent Types (often defined by absence of above or unique structure)
        'nitro': '[NX3](=O)=O',             # e.g., Nitromethane (often treated as a distinct solvent type, lower priority FG)
        'aromatic': '[ar]',                 # e.g., Benzene, Toluene (Indicates presence of an aromatic ring system)
        'cycloalkane': '[CX4R]',            # e.g., Cyclohexane (Indicates a saturated carbon in a ring)
        'alkane': '[CX4]',                  # e.g., Hexane (Indicates a saturated carbon - very general pattern)
    }
    
    for group_name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            classifications.append(group_name)
    
    # Add basic molecular properties
    if rdMolDescriptors.CalcNumRings(mol) > 0:
        classifications.append('cyclic')
    if mol.GetNumAtoms() > 20:
        classifications.append('large_molecule')
    
    # If no specific groups found, classify as hydrocarbon
    if not classifications:
        classifications.append('hydrocarbon')
    
    return classifications

def classify_solvent(smiles):
    """
    Classify solvent based on priority order:
    1. Water
    2. O,N-containing
    3. S-containing
    4. X(halogen)-containing
    5. Aromatics
    6. Cycloalkanes
    7. Hydrocarbons
    8. P-containing
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ['invalid_smiles']
    
    # Special case for water
    if smiles == 'O':
        return ['Water']
    
    # Define SMARTS patterns in priority order
    solvent_types = [
        ('Water', '[OH2]'),
        ('O,N-containing', '[$([OH1,OH0]),$([NH2,NH1,NH0,N+,n])]'),
        ('S-containing', '[#16]'),
        ('X-containing', '[F,Cl,Br,I]'),
        ('Aromatics', '[ar]'),
        ('P-containing', '[#15]')
    ]
    
    # Check each category in order
    for category, smarts in solvent_types:
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            return [category]
    
    # Check for cycloalkanes (after aromatics check)
    if rdMolDescriptors.CalcNumRings(mol) > 0 and not rdMolDescriptors.CalcNumAromaticRings(mol) > 0:
        return ['Cycloalkanes']
    
    # Default to hydrocarbon if no other categories match
    return ['Hydrocarbons']

def classify_solvent_1(smiles):
    """
    Classify solvent based on chemical classes with priority order:
    1. Water
    2. Benzenoids & Organo-N-S Compounds
    3. Hydrocarbons
    4. Lipids and Lipid-like Molecules
    5. Organic Acids and Derivatives
    6. Organic Oxygen Compounds
    7. Organohalogen Compounds
    8. Organoheterocyclic Compounds
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ['invalid_smiles']
    
    # Define SMARTS patterns for each category
    categories = [
        ('Water', '[OH2]'),
        ('Benzenoids & Organo-N-S Compounds', 
         '[$([ar6]:[ar6]),$([#7,#16;!$(S=O);!$(S(=O)=O)])]'),
        ('Organohalogen Compounds', '[F,Cl,Br,I]'),
        ('Organic Acids and Derivatives', 
         '[$([CX3](=O)[OH]),$([CX3](=O)[O-]),$([CX3](=O)[OX2H0])]'),
        ('Organic Oxygen Compounds', 
         '[$([OH1,OH0]),$([#6][OX2][#6])]'),
        ('Organoheterocyclic Compounds', 
         '[$([r3,r4,r5,r6,r7,r8]~[#7,#8,#16,#15])]'),
        ('Lipids and Lipid-like Molecules',
         '[$([CX4H2][CX4H2][CX4H2]),$([CH3][CX4H2][CX4H2][CX4H2][CH3])]')
    ]
    
    # Special case for water
    if smiles == 'O':
        return ['Water']
    
    # Check each category in order
    for category, smarts in categories:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return [category]
    
    # Default to hydrocarbons if no other categories match
    return ['Hydrocarbons']


def smiles_index_mapping(smiles_list):
    unique_smiles = {}
    index = 1

    for i, smiles in enumerate(smiles_list): 
        if smiles not in unique_smiles:
            molecule_info = {
                'solute_index': f"mol_{index:04d}",
                'solute_type': classify_molecule(smiles)
            }
            unique_smiles[smiles] = molecule_info
            index += 1
    return unique_smiles

def get_dataset_info(data_dir):
    """Load dataset info from CSV file"""
    df = pd.read_csv(data_dir / "smiles.csv", delimiter=";", dtype=str, comment=None)
    
    info_dict = {}
    for _, row in df.iterrows():
        info_dict[int(row['id'])] = {
            "key": row['mol_id'],
            "smile": row['solute_smiles'],
            "solv_id": row['solvent_id'],
            "solv_smi": row['solvent_smiles'],
            "properties": {
                "data_origin": row['data_origin'],
                "solvent_name": row['solvent_name'],
                "solvation_free_energy": float(row['solvation_free_energy']) if row['solvation_free_energy'] != '' else None,
            }
        }
    return info_dict

def scale_dataset(dataset, scale_R, scale_U, fractional=True):
    """Scales the dataset."""

    # Make large enough to fit individual structures
    box = 10 * (dataset["R"].max(axis=(1, 2)) - dataset["R"].min(axis=(1, 2)))

    if fractional:
        dataset['R'] = dataset['R'] / box[:, onp.newaxis, onp.newaxis]
    else:
        dataset['R'] = dataset['R'] * scale_R

    dataset['dF'] = dataset['dF'] * scale_U
    dataset['box'] = scale_R * box[:, onp.newaxis, onp.newaxis] * onp.tile(onp.eye(3), (box.size, 1, 1))

    return dataset

if __name__=="__main__":
    data, info = download_combined("/home/linying/Datasets", split="stratified", regenerate=True)
