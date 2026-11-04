import haiku as hk

from . import ScaleShiftLayer


class AtomicEnergyLayer(hk.Module):
    """Adds atomic energy to the model."""

    def __init__(self, num_species: int, scale: float = 1.0, shift: float = 0.0):
        super().__init__()

        self.scale_shift = ScaleShiftLayer(scale=scale, shift=shift)
        self.atomic_energies = hk.Embed(num_species, 1)

    def __call__(self, per_atom_energies, species):
        atomic_energies, = self.atomic_energies(species).T
        return atomic_energies + self.scale_shift(per_atom_energies)
