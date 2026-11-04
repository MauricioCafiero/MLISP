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


import jax.tree_util as tree_util

import sys
sys.path.append("/ds/project/linying/Datasets/solvatum/solvatum")
from ui import Database

def download_explicit_testset(root="./_data",
                      scale_R=1.0,
                      scale_U=4.184,
                      fractional=True,
                      subsets=None,
                      filter=True,
                      **kwargs):
    dataset, info = download_and_process(root=root, subsets=subsets)
    dataset = scale_dataset(dataset, scale_R, scale_U=1.0, fractional=fractional)
    if filter:
        dataset = filter_dataset(dataset)
    dataset = {"testing": dataset}

    return dataset, info

def filter_dataset(dataset):
    data_solvatum = Database()
    data_solvatum.energy_unit = 'kcal/mol'
    data_testset = pd.read_csv("/ds/project/linying/Datasets/explicit_testset/ci6b00081_si_002.csv", sep=",")

    count = 0
    index_list = []
    for i in range(len(data_testset.iloc[1:])):
        try:
            solvent = data_testset.iloc[i+1]["solvent"].lower().replace("_", " ")
            solute = data_testset.iloc[i+1]["solute"].lower().replace("_", " ")
            if str(solvent) == 'nn-dimethylacetamide':
                solvent = 'n,n-dimethylacetamide'

            if str(solute) == 'nn-dimethylacetamide':
                solute = 'n,n-dimethylacetamide'
            
            if str(solute) == 'nn-dimethylformamide':
                solute = 'dimethylformamide'
            if str(solvent) == 'nn-dimethylformamide':
                solvent = 'dimethylformamide'

            # solvent = data_testset.iloc[i+1]["solvent"]
            # solute = data_testset.iloc[i+1]["solute"]
            # print(f"solvent: {solvent}, solute: {solute}")
            data = data_solvatum.filtering(solvent=solvent, solute=solute)
            count += 1
            # print(f"solvent: {solvent}, solute: {solute}, value: {data['deltaG_solv']*4.184}, {data_testset.iloc[i+1]['Exper.']}")
            
        except :
            # print(f"No matched data")
            index_list.append(i)    
    print(f"matched data: {count}")

    dataset = tree_util.tree_map(lambda x: x[onp.array(index_list)], dataset)
    return dataset

def download_and_process(root, subsets=None):
    data_dir = Path(root)
    data_dir.mkdir(exist_ok=True, parents=True)
    data_dir = data_dir / "explicit_testset"

    # if (data_dir / "database.npz").exists():
    #     dataset = dict(onp.load(data_dir / "database.npz"))
    #     info = get_dataset_info(data_dir)
    #     return dataset, info
    
    conf_dir = data_dir / "confs"
    conf_dir.mkdir(exist_ok=True)

    raw_data = {
        "R": [],
        "dF": [],
        "species": [],
        "mass": [],
        "bonds": [],
        "solv_species": [],
        "task": []
    }

    info = []
    df = download_raw(data_dir)

    # Process each unique solute
    unique_solutes = df[['solute_smiles']].drop_duplicates()
    # Define allowed atomic numbers
    allowed_atomic_numbers = {1, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 35, 53}
    for i, row in unique_solutes.iterrows():
        mol_id = i
        solute_smiles = row['solute_smiles']
        
        # Generate 3D conformer if not exists
        if not (conf_dir / f"mol_{mol_id}.pdb").exists():
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

            info.append(data_point)
    # Save SMILES mapping with all data point information
    with open(data_dir / "smiles.csv", "w") as f:
        # Write header
        header = ["id", "solute_smiles", "solvent_id", "solvent_name", 
                "data_origin", "TI", "COSMO-RS", "solute_name"]
        f.write(";".join(header) + "\n")
        
        # Write data points
        for idx, data_point in enumerate(info):
            row = [
                str(idx),
                data_point['solute_smiles'],
                str(data_point['solvent_index']),
                data_point['solvent_name'],
                str(data_point['solvation_free_energy']),
                str(data_point['TI']),
                str(data_point['COSMO-RS']),
                data_point['solute_name']
            ]
            f.write(";".join(row) + "\n")
    
    # Process raw data with padding
    max_atoms = max([len(species) for species in raw_data["species"]] + [86])
    max_bonds = max([bonds.shape[1] for bonds in raw_data["bonds"]] + [170])
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
    }
    
    # Fill the padded arrays
    for idx, entry in enumerate(zip(*raw_data.values())):
        pos, free_eng, species, mass, bonds, solv_species, task_num = entry
        
        data["R"][idx, :mass.size] = pos
        data["dF"][idx] = free_eng
        data["mask"][idx, :mass.size] = True
        data["species"][idx, :mass.size] = species
        data["mass"][idx, :mass.size] = mass
        data["bonds"][idx, :, :bonds.shape[1]] = bonds
        data["solv_species"][idx, :mass.size] = solv_species
        data["task"][idx] = task_num
    
    # Save processed data
    onp.savez(data_dir / "database.npz", **data)
    
    return data, get_dataset_info(data_dir)

def download_raw(data_dir):
    data_testset = pd.read_csv("/ds/project/linying/Datasets/explicit_testset/ci6b00081_si_002.csv", sep=",")
    solvent_index = json.load(open("/ds/project/linying/Datasets/solvatum/solvent_index.json", "r"))
    solvent_index_low = {k.lower(): v for k, v in solvent_index.items()}
    data_points = []
    for i in range(len(data_testset.iloc[1:])):
        solvent = data_testset.iloc[i+1]["solvent"].lower().replace("_", " ")
        solute = data_testset.iloc[i+1]["solute"].lower().replace("_", " ")
        if str(solvent) == 'nn-dimethylacetamide':
            solvent = 'n,n-dimethylacetamide'

        if str(solute) == 'nn-dimethylacetamide':
            solute = 'n,n-dimethylacetamide'
        
        if str(solute) == 'nn-dimethylformamide':
            solute = 'dimethylformamide'
        if str(solvent) == 'nn-dimethylformamide':
            solvent = 'dimethylformamide'
        print(solute)
        database_solvatum = Database()
        database_solvatum.energy_unit='kcal/mol'
        
        try:
            smiles = database_solvatum.get_molecule_properties(solute)["SMILES"]
        except:
            url = 'http://cactus.nci.nih.gov/chemical/structure/' + parse.quote(solute) + '/smiles'
            smiles = request.urlopen(url).read().decode('utf8')
            # print(f"Failed to fetch SMILES for {solute}. Skipping.")
            # continue


        print(smiles)
        try:
            data_point = {
                'solvent_name': solvent,
                'solute_name': solute,
                'solute_smiles': smiles,
                'solvent_index': solvent_index_low[solvent]-1,
                'solvation_free_energy': data_testset.iloc[i+1]['Exper.'],
                'TI': data_testset.iloc[i+1]['TI'],
                'COSMO-RS': data_testset.iloc[i+1]['COSMO-RS.3'],
            }
        except KeyError:
            print(f"Solvent {solvent} not found in solvent index. Skipping.")
            continue
        data_points.append(data_point)

        # if i==5:
        #     break
    df = pd.DataFrame(data_points)
    # print(df)
    return df

def get_dataset_info(data_dir):
    """Load dataset info from CSV file"""
    df = pd.read_csv(data_dir / "smiles.csv", delimiter=";", dtype=str, comment=None)
    
    info_dict = {}
    for _, row in df.iterrows():
        info_dict[int(row['id'])] = {
            "key": row['solute_smiles'],
            "solv_id": row['solvent_id'],
            "TI": row["TI"],
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

if __name__ == "__main__":
    dataset, info = download_explicit_testset(
        root="/ds/project/linying/Datasets/",
        scale_R=1.0,
        scale_U=4.184,
        fractional=False,
        subsets=None,
    )