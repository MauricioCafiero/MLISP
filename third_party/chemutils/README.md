# Chemutils

Chemutils contains potential models, datasets, and tools useful for projects of
the MFM group.


## Installation

To only use the package first clone it and then install it via pip:
```bash
git clone git@gitlab.lrz.de:mfm/science/chemutils.git ./chemutils
pip install -e ./chemutils
```

## Datasets

### Titanium

Curated dataset of DFT calculations of bulk titanium. The dataset contains
equilibrium and strained boxes of titanium, each containing 256 atoms.

Usage:
```python
from chemutils.datasets import titanium

# Automatically downloads the dataset to a specified root location
dataset = titanium.download_dataset(root=".")

# The dataset is a dictionary with three spilts "training", "validation", and
# "testing". Each split contains a dictionary with the keys:
#    "R": The positions of the atoms
#    "F": The forces acting on the atoms
#    "U": The potential energy of the system
#    "box": The box vectors of the system
#    "virial": The virial tensor of the system
#    "virial_weights": Weights that are zero for virials from non-bulk systems
```

The dataset is subject to the following example:
[Bottom-Up and Top-Down Training of Atomistic Titanium](https://chemtrain.readthedocs.io/en/latest/examples/AT_titanium_fused_training.html)


## Models

### MACE

A modified version of the MACE potential implemented in JAX.
The original code can be found [here](https://github.com/ACEsuit/mace-jax).

