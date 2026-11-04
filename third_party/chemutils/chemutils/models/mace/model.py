import functools
from copy import deepcopy

import math
from typing import Callable, Optional, Union, Set
from collections import OrderedDict

import numpy as onp

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

import jax.nn as jax_nn

from jax_md import space, partition, nn, util as md_util

from typing import Tuple, Any, Callable

from jax_md_mod.model import sparse_graph
# from jax_md_mod import custom_electrostatics

from .utils import safe_norm
from .layers import (
    EquivariantProductBasisLayer,
    InteractionLayer,
    LinearNodeEmbeddingLayer,
    LinearNodeEmbeddingBlock,
    LinearNodeEmbeddingVisual,
    LinearReadoutLayer,
    NonLinearReadoutLayer, 
    ScaleShiftLayer,
    OutputSolventLayer
)
from .test_block import FiLMLayer_e3nn, FiLMLayer_e3nn1, OutputSolventEmbedding
from jax_md_mod.model.layers import (
    RadialBesselLayer
)

from chemutils.models.layers import AtomicEnergyLayer

from collections.abc import Sequence
def shift_init(shape: Sequence[int], dtype: Any, **kwargs) -> jax.Array:
    return jax.numpy.zeros(shape)

class MACE_shift(hk.Module):
    """Per species potential shift for MACE model.

    Needed in order to fit pretrained implicit solvent model to free energy data in ReSolv."""

    def __init__(self, unique_species, species_indep_shift=False):
        """Initializes per species potential shift for MACE model

        Args:
            unique_species: Array of all unique species in the model will be
                trained on.
            species_indep_shift: Whether to apply species independent shift. Per
                default only per species shift is applied.
        """
        super().__init__()

        self._unique_species = unique_species
        # Number of different atom species the network is supposed to process
        self._max_species = int(onp.max(onp.asarray(unique_species))) + 1
        # Number of unique species
        self._n_unique_species = int(unique_species.shape[0]-1)
        # constant species independent potential shift
        self._species_indep_shift = species_indep_shift

        # number of trainable parameters
        n_params = self._n_unique_species
        if species_indep_shift:
            n_params += 1
        self._per_species_shift = hk.get_parameter(
            'Per_species_energy_shift', [n_params], init=shift_init)

    def params_energy_shift_map(self, species):
        # chemtrain processes species as indices not atomic numbers (atomic number needed for this calculation)
        species = species + jnp.ones(species.shape)

        # map from parameters to all atom types
        _t1 = jnp.transpose(jax.nn.one_hot(self._unique_species, self._max_species))
        # map from atom types to species fo configuration
        _t2 = jax.nn.one_hot(species, self._max_species)
        _t = jnp.matmul(_t2, _t1)
        params_energy_shift_map = jnp.sum(_t, axis=0)
        if self._species_indep_shift:
            return params_energy_shift_map.at[0].set(1)
        else:
            return params_energy_shift_map[1:]

    def __call__(self, species: jnp.ndarray):
        params_energy_shift_map = self.params_energy_shift_map(species)
        # calculate per species energy shift
        dU = jnp.vecdot(params_energy_shift_map, self._per_species_shift)
        return dU




class MACE(hk.Module):
    def __init__(
        self,
        *,
        output_irreps: e3nn.Irreps,  # Irreps of the output, default 1x0e
        num_interactions: int,  # Number of interactions (layers), default 2
        hidden_irreps: e3nn.Irreps,  # 256x0e or 128x0e + 128x1o
        readout_mlp_irreps: e3nn.Irreps,  # Hidden irreps of the MLP in last readout, default 16x0e
        avg_num_neighbors: float,
        num_species: int,
        num_features: int = None,  # Number of features per node, default gcd of hidden_irreps multiplicities
        n_radial_basis: int = 8,  # Number of radial basis functions
        envelope_p: int = 6,  # Order of the envelope polynomial
        max_ell: int = 3,  # Max spherical harmonic degree, default 3
        epsilon: Optional[float] = None,
        correlation: int = 3,  # Correlation order at each layer (~ node_features^correlation), default 3
        activation: Callable = jax.nn.silu,  # activation function
        gate: Callable = jax.nn.sigmoid,  # gate function
        soft_normalization: Optional[float] = None,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
        interaction_irreps: Union[str, e3nn.Irreps] = "o3_restricted",  # or o3_full
        skip_connection_first_layer: bool = False,
        embed_dim: int = 32,
    ):
        super().__init__()

        output_irreps = e3nn.Irreps(output_irreps)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)

        if num_features is None:
            self.num_features = functools.reduce(
                math.gcd, (mul for mul, _ in hidden_irreps)
            )
            self.hidden_irreps = e3nn.Irreps(
                [(mul // self.num_features, ir) for mul, ir in hidden_irreps]
            )
        else:
            self.num_features = num_features
            self.hidden_irreps = hidden_irreps

        if interaction_irreps == "o3_restricted":
            self.interaction_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        elif interaction_irreps == "o3_full":
            self.interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(max_ell))
        else:
            self.interaction_irreps = e3nn.Irreps(interaction_irreps)

        self.correlation = correlation
        self.epsilon = epsilon
        self.readout_mlp_irreps = readout_mlp_irreps
        self.activation = activation
        self.gate = gate
        self.num_interactions = num_interactions
        self.output_irreps = output_irreps
        self.num_species = num_species
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal
        self.max_ell = max_ell
        self.soft_normalization = soft_normalization
        self.skip_connection_first_layer = skip_connection_first_layer

        epsilon = hk.get_parameter(
            "varepsilon", shape=(),
            init=hk.initializers.Constant(jnp.sqrt(avg_num_neighbors))
        )
        self.epsilon = 1 / jnp.sqrt(1 + epsilon ** 2)

        # Embeddings
        self.node_embedding = LinearNodeEmbeddingLayer(
            embed_dim, self.num_species, self.num_features * self.hidden_irreps
        )
        self.radial_embedding = RadialBesselLayer(
            cutoff=1.0, num_radial=n_radial_basis, envelope_p=envelope_p
        )

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
        is_training: bool = False,
    ) -> e3nn.IrrepsArray:
        assert vectors.ndim == 2 and vectors.shape[1] == 3
        assert node_species.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        if node_mask is None:
            node_mask = jnp.ones(node_species.shape[0], dtype=jnp.bool_)

        # Embeddings
        node_feats = self.node_embedding(node_species).astype(vectors.dtype)  # [n_nodes, feature * irreps]
        rbf = self.radial_embedding(safe_norm(vectors.array, axis=-1))

        # Interactions
        outputs = []
        node_outputs = None
        for i in range(self.num_interactions):
            first = i == 0
            last = i == self.num_interactions - 1

            hidden_irreps = (
                self.hidden_irreps
                if not last
                else self.hidden_irreps.filter(self.output_irreps)
            )

            node_outputs, node_feats = MACELayer(
                first=first,
                last=last,
                num_features=self.num_features,
                interaction_irreps=self.interaction_irreps,
                hidden_irreps=hidden_irreps,
                max_ell=self.max_ell,
                epsilon=self.epsilon,
                activation=self.activation,
                gate=self.gate,
                num_species=self.num_species,
                correlation=self.correlation,
                output_irreps=self.output_irreps,
                readout_mlp_irreps=self.readout_mlp_irreps,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                soft_normalization=self.soft_normalization,
                skip_connection_first_layer=self.skip_connection_first_layer,
                name=f"layer_{i}",
            )(
                vectors,
                node_feats,
                node_species,
                rbf,
                senders,
                receivers,
                node_mask,
                is_training
            )
            outputs += [node_outputs]  # list of [n_nodes, output_irreps]

        # TODO: Not sure whether outputs from all layers are important.
        return node_outputs # e3nn.stack(outputs, axis=1)  # [n_nodes, num_interactions, output_irreps]

class MACE_CONSOL(hk.Module):
    def __init__(
        self,
        *,
        output_irreps: e3nn.Irreps,  # Irreps of the output, default 1x0e
        num_interactions: int,  # Number of interactions (layers), default 2
        hidden_irreps: e3nn.Irreps,  # 256x0e or 128x0e + 128x1o
        readout_mlp_irreps: e3nn.Irreps,  # Hidden irreps of the MLP in last readout, default 16x0e
        avg_num_neighbors: float,
        num_species: int,
        num_features: int = None,  # Number of features per node, default gcd of hidden_irreps multiplicities
        n_radial_basis: int = 8,  # Number of radial basis functions
        envelope_p: int = 6,  # Order of the envelope polynomial
        max_ell: int = 3,  # Max spherical harmonic degree, default 3
        epsilon: Optional[float] = None,
        correlation: int = 3,  # Correlation order at each layer (~ node_features^correlation), default 3
        activation: Callable = jax.nn.silu,  # activation function
        gate: Callable = jax.nn.sigmoid,  # gate function
        soft_normalization: Optional[float] = None,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
        interaction_irreps: Union[str, e3nn.Irreps] = "o3_restricted",  # or o3_full
        skip_connection_first_layer: bool = False,
        embed_dim: int = 32,
        embed_solvent_dim: int = 32,
        num_solvent_species: int,
        solvent_interaction: str = "input",  # or "output"
        nonlinear_solembed: bool = False,
        block_name: Optional[str] = None,
        solv_tp: bool = False,
        heads: int = 8,
    ):
        super().__init__()

        output_irreps = e3nn.Irreps(output_irreps)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)

        if num_features is None:
            self.num_features = functools.reduce(
                math.gcd, (mul for mul, _ in hidden_irreps)
            )
            self.hidden_irreps = e3nn.Irreps(
                [(mul // self.num_features, ir) for mul, ir in hidden_irreps]
            )
        else:
            self.num_features = num_features
            self.hidden_irreps = hidden_irreps

        if interaction_irreps == "o3_restricted":
            self.interaction_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        elif interaction_irreps == "o3_full":
            self.interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(max_ell))
        else:
            self.interaction_irreps = e3nn.Irreps(interaction_irreps)

        self.correlation = correlation
        self.epsilon = epsilon
        self.readout_mlp_irreps = readout_mlp_irreps
        self.activation = activation
        self.gate = gate
        self.num_interactions = num_interactions
        self.output_irreps = output_irreps
        self.num_species = num_species
        self.num_solvent_species = num_solvent_species
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal
        self.max_ell = max_ell
        self.soft_normalization = soft_normalization
        self.skip_connection_first_layer = skip_connection_first_layer
        self.solvent_interaction = solvent_interaction
        self.solv_tp = solv_tp

        epsilon = hk.get_parameter(
            "varepsilon", shape=(),
            init=hk.initializers.Constant(jnp.sqrt(avg_num_neighbors))
        )
        self.epsilon = 1 / jnp.sqrt(1 + epsilon ** 2)

        # Embeddings
        if solvent_interaction == "input":
            self.node_embedding = LinearNodeEmbeddingBlock(
                embed_dim, self.num_species, self.num_features * self.hidden_irreps, 
                embed_solvent_dim, self.num_solvent_species, nonlinear_solembed, block_name, heads=heads
            )
        elif solvent_interaction == "output":
            self.node_embedding = LinearNodeEmbeddingLayer(
                embed_dim, self.num_species, self.num_features * self.hidden_irreps
            )

        self.radial_embedding = RadialBesselLayer(
            cutoff=1.0, num_radial=n_radial_basis, envelope_p=envelope_p
        )

        if solv_tp:
            self.FiLMLayer = FiLMLayer_e3nn(self.output_irreps)

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        solvent_specie: jnp.ndarray,
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
        is_training: bool = False,
    ) -> e3nn.IrrepsArray:
        assert vectors.ndim == 2 and vectors.shape[1] == 3
        assert node_species.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        if node_mask is None:
            node_mask = jnp.ones(node_species.shape[0], dtype=jnp.bool_)

        # Embeddings
        if self.solvent_interaction == "input":
            if self.solv_tp:
                node_feats, solv_feats = self.node_embedding(node_species, solvent_specie, solv_tp = self.solv_tp)  # [n_nodes, feature * irreps]
                node_feats = node_feats.astype(vectors.dtype)  # [n_nodes, feature * irreps]
                solv_feats = solv_feats.astype(vectors.dtype)  # [n_nodes, solv_embed_dim * 0e]
                print("[MACE_CONSOL] add solvent residual connection.")
            else:
                node_feats = self.node_embedding(node_species, solvent_specie)
                solv_feats = None
                print("[MACE_CONSOL] without solvent residual connection.")
        elif self.solvent_interaction == "output":
            print("[MACE_CONSOL] Use output solvent embedding.")
            node_feats = self.node_embedding(node_species).astype(vectors.dtype)  # [n_nodes, feature * irreps]
            solv_feats = None

        rbf = self.radial_embedding(safe_norm(vectors.array, axis=-1))

        # Interactions
        outputs = []
        node_outputs = None
        for i in range(self.num_interactions):
            first = i == 0
            last = i == self.num_interactions - 1

            hidden_irreps = (
                self.hidden_irreps
                if not last
                else self.hidden_irreps.filter(self.output_irreps)
            )

            
            node_outputs, node_feats = MACELayer(
                first=first,
                last=last,
                num_features=self.num_features,
                interaction_irreps=self.interaction_irreps,
                hidden_irreps=hidden_irreps,
                max_ell=self.max_ell,
                epsilon=self.epsilon,
                activation=self.activation,
                gate=self.gate,
                num_species=self.num_species,
                correlation=self.correlation,
                output_irreps=self.output_irreps,
                readout_mlp_irreps=self.readout_mlp_irreps,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                soft_normalization=self.soft_normalization,
                skip_connection_first_layer=self.skip_connection_first_layer,
                solv_tp = self.solv_tp,
                name=f"layer_{i}",
            )(
                vectors,
                node_feats,
                node_species,
                rbf,
                senders,
                receivers,
                node_mask,
                is_training,
                solv_feats=solv_feats,
            )
            outputs += [node_outputs]  # list of [n_nodes, output_irreps]

        # modulated output with film layer
        # gamma, beta = self.FiLMLayer(solv_feats) # [out]
        # node_outputs = node_outputs * (gamma + 1) + beta

        # TODO: Not sure whether outputs from all layers are important.
        return node_outputs # e3nn.stack(outputs, axis=1)  # [n_nodes, num_interactions, output_irreps]


class MACELayer(hk.Module):
    def __init__(
        self,
        *,
        first: bool,
        last: bool,
        num_features: int,
        interaction_irreps: e3nn.Irreps,
        hidden_irreps: e3nn.Irreps,
        activation: Callable,
        gate: Callable,
        num_species: int,
        epsilon: Optional[float],
        name: Optional[str],
        # InteractionBlock:
        max_ell: int,
        # EquivariantProductBasisBlock:
        correlation: int,
        symmetric_tensor_product_basis: bool,
        off_diagonal: bool,
        soft_normalization: Optional[float],
        # ReadoutBlock:
        output_irreps: e3nn.Irreps,
        readout_mlp_irreps: e3nn.Irreps,
        skip_connection_first_layer: bool = False,
        num_solv_species: int = 148,
        solv_tp: bool = False,
    ) -> None:
        super().__init__(name=name)

        # self.dropout = e3nn.haiku.Dropout(p=0.5)

        self.first = first
        self.last = last
        self.num_features = num_features
        self.interaction_irreps = interaction_irreps
        self.hidden_irreps = hidden_irreps
        self.max_ell = max_ell
        self.activation = activation
        self.gate = gate
        self.num_species = num_species
        self.num_solvent_species = num_solv_species
        self.epsilon = epsilon
        self.correlation = correlation
        self.output_irreps = output_irreps
        self.readout_mlp_irreps = readout_mlp_irreps
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal
        self.soft_normalization = soft_normalization
        self.skip_connection_first_layer = skip_connection_first_layer
        if solv_tp:
            self.film_layer = FiLMLayer_e3nn1(feature_dim=num_features * hidden_irreps)

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
        is_training: bool = False,
        solv_feats: e3nn.IrrepsArray = None,  # [n_nodes, irreps],
    ):
        if node_mask is None:
            node_mask = jnp.ones(node_specie.shape[0], dtype=jnp.bool_)

        sc = None
        if not self.first or self.skip_connection_first_layer:
            sc = e3nn.haiku.Linear(
                self.num_features * self.hidden_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp",
            )(
                node_specie, node_feats
            )  # [n_nodes, feature * hidden_irreps]

            # print(f"Use dropout for hidden irreps")
            # sc = self.dropout(hk.next_rng_key(), sc, is_training=is_training)

        node_feats = InteractionLayer(
            target_irreps=self.num_features * self.interaction_irreps,
            epsilon=self.epsilon,
            max_ell=self.max_ell,
            activation=self.activation,
        )(
            vectors=vectors,
            node_feats=node_feats,
            radial_embedding=radial_embedding,
            receivers=receivers,
            senders=senders,
            solvent_embedding=solv_feats[0] if solv_feats is not None else None,
        )

        node_feats *= self.epsilon

        if self.first:
            # Selector TensorProduct
            node_feats = e3nn.haiku.Linear(
                self.num_features * self.interaction_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp_first",
            )(node_specie, node_feats)

        node_feats = EquivariantProductBasisLayer(
            target_irreps=self.num_features * self.hidden_irreps,
            correlation=self.correlation,
            num_species=self.num_species,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )(node_feats=node_feats, node_specie=node_specie)

        # node_feats = self.dropout(hk.next_rng_key(), node_feats, is_training=is_training)

        if self.soft_normalization is not None:

            def phi(n):
                n = n / self.soft_normalization
                return 1.0 / (1.0 + n * e3nn.sus(n))

            node_feats = e3nn.norm_activation(
                node_feats, [phi] * len(node_feats.irreps)
            )

        # if solv_feats is not None:
            # gamma, beta = self.film_layer(solv_feats)
            # tensor_produc = e3nn.tensor_product(node_feats, gamma)
            # # reduced_tensor_produc = e3nn.haiku.Linear(node_feats.irreps, parameter_initializer=lambda: hk.initializers.TruncatedNormal(stddev=1e-3))(
            # #     tensor_produc
            # # )

            # # node_feats = reduced_tensor_produc + node_feats
            # scalar_feats = node_feats.filter("0e")
            # vector_feats = node_feats.filter("1o")      
            # modulated_scalar = scalar_feats * (1 + gamma) + beta   
            # node_feats = e3nn.concatenate(
            #     [modulated_scalar, vector_feats])
            # node_feats_array = node_feats.array * gamma + beta
            # node_feats = e3nn.IrrepsArray(node_feats.irreps, node_feats_array)

        if sc is not None:
            node_feats = node_feats + sc  # [n_nodes, feature * hidden_irreps]

        if not self.last:
            node_outputs = LinearReadoutLayer(self.output_irreps)(
                node_feats
            )  # [n_nodes, output_irreps]
        else:  # Non linear readout for last layer
            node_outputs = NonLinearReadoutLayer(
                self.readout_mlp_irreps,
                self.output_irreps,
                activation=self.activation,
                gate=self.gate
            )(
                node_feats
            )  # [n_nodes, output_irreps]

        return node_outputs, node_feats


mace_default_kwargs = OrderedDict(
    embed_dim = 32,
    input_irreps = "1o",
    output_irreps = "1x0e",
    hidden_irreps = "128x0e + 128x1o",
    readout_irreps = "16x0e",
    num_interactions = 2,
    mlp_n_hidden=64,
    mlp_n_layers=2,
    max_ell = 2,
    avg_num_neighbors = 1,
    n_radial_basis = 8,
    envelope_p = 6,
    correlation = 3,
    block_name = None
)


def mace_neighborlist_pp(displacement: space.DisplacementFn,
                         r_cutoff: float,
                         n_species: int = 100,
                         positions_test: jnp.ndarray = None,
                         neighbor_test: partition.NeighborList = None,
                         max_edge_multiplier: float = 1.25,
                         max_edges=None,
                         avg_num_neighbors: float = None,
                         mode: str = "energy",
                         per_particle: bool = False,
                         positive_species: bool = False,
                         **mace_kwargs
                         ) -> Tuple[nn.InitFn, Callable[[Any, md_util.Array],
                                                               md_util.Array]]:
    """NequIP model for property prediction.

    Args:
        displacement: Jax_md displacement function
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list
        n_species: Number of different atom species the network is supposed
            to process.
        positions_test: Sample positions to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        neighbor_test: Sample neighborlist to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        max_edge_multiplier: Multiplier for initial estimate of maximum edges.
        max_edges: Expected maximum of valid edges.
        avg_num_neighbors: Average number of neighbors per particle. Guessed
            if positions_test and neighbor_test are provided.
        mode: Prediction mode of the model. If "property_prediction" (default),
            returns the learned node features. If "energy_prediction", returns
            the total energy of the system.
        per_particle: Return per-particle energies instead of total energy.
        positive_species: True if the smallest occurring species is 1, e.g., in
            case of atomic numbers.
        mace_kwargs: Kwargs to change the default structure of NequIP.
            For definition of the kwargs, see NequIP.


    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters. The energy function requires the same input as
        other energy functions with neighbor lists in jax_md.energy.
    """
    kwargs = deepcopy(mace_default_kwargs)
    kwargs.update(mace_kwargs)

    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    # Checking only necessary if neighbor list is dense
    _avg_num_neighbors = None
    if positions_test is not None and neighbor_test is not None:
        if neighbor_test.format == partition.Dense:
            print('Capping edges and triplets. Beware of overflow, which is'
                  ' currently not being detected.')

            testgraph, _ = sparse_graph.sparse_graph_from_neighborlist(
                displacement, positions_test, neighbor_test, r_cutoff)
            max_edges = jnp.int32(jnp.ceil(testgraph.n_edges * max_edge_multiplier))

            # cap maximum edges and angles to avoid overflow from multiplier
            n_particles, n_neighbors = neighbor_test.idx.shape
            max_edges = min(max_edges, n_particles * n_neighbors)

            print(f"Estimated max. {max_edges} edges.")

            _avg_num_neighbors = testgraph.n_edges / n_particles
        else:
            n_particles = neighbor_test.idx.shape[0]
            _avg_num_neighbors = onp.sum(neighbor_test.idx[0] < n_particles)
            _avg_num_neighbors /= n_particles

    if avg_num_neighbors is None:
        avg_num_neighbors = _avg_num_neighbors
    assert avg_num_neighbors is not None, (
        "Average number of neighbors not set and no test graph was provided."
    )

    @hk.without_apply_rng
    @hk.transform
    def model(position: md_util.Array,
              neighbor: partition.NeighborList,
              species: md_util.Array = None,
              mask: md_util.Array = None,
              **dynamic_kwargs):
        if species is None:
            print(f"[MACE] Use default species")
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        elif positive_species:
            species -= 1
        if mask is None:
            print(f"[MACE] Use default mask")
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        # Compute the displacements for all edges
        dyn_displacement = functools.partial(displacement, **dynamic_kwargs)

        if neighbor.format == partition.Dense:
            graph, _ = sparse_graph.sparse_graph_from_neighborlist(
                dyn_displacement, position, neighbor, r_cutoff,
                species, max_edges=max_edges, species_mask=mask
            )
            senders = graph.idx_i
            receivers = graph.idx_j
        else:
            assert neighbor.idx.shape == (2, neighbor.idx.shape[1]), "Neighbor list has wrong shape."
            senders, receivers = neighbor.idx

        invalid_idx = position.shape[0]

        # Set invalid edges to the cutoff to avoid numerical issues
        vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
        if neighbor.format != partition.Dense:
            valid_edges = jnp.logical_and(
                senders < invalid_idx,
                receivers < invalid_idx
            )
            distances = space.distance(vectors)
            mace_edges = jnp.logical_and(valid_edges, distances < r_cutoff)
            senders = jnp.where(mace_edges, senders, invalid_idx)
            receivers = jnp.where(mace_edges, receivers, invalid_idx)
        vectors = jnp.where(
            jnp.logical_and(
                senders < invalid_idx,
                receivers < position.shape[0]
            )[:, jnp.newaxis], vectors, r_cutoff / jnp.sqrt(3))
        vectors /= r_cutoff

        # Sort vectors by length and remove up to max_edges edges
        # lengths = jnp.linalg.norm(vectors, axis=-1)
        # sort_idx = jnp.argsort(lengths)
        # vectors = vectors[sort_idx][:max_edges]
        # senders = senders[sort_idx][:max_edges]
        # receivers = receivers[sort_idx][:max_edges]

        vectors = e3nn.IrrepsArray(
            e3nn.Irreps(kwargs['input_irreps']), vectors
        )

        net = MACE(
            output_irreps=kwargs["output_irreps"],
            num_interactions=kwargs["num_interactions"],
            hidden_irreps=kwargs["hidden_irreps"],
            readout_mlp_irreps=kwargs["readout_irreps"],
            avg_num_neighbors=avg_num_neighbors,
            num_species=n_species,
            max_ell=kwargs["max_ell"],
            n_radial_basis=kwargs["n_radial_basis"],
            envelope_p=kwargs["envelope_p"],
            correlation=kwargs["correlation"],
            embed_dim=kwargs["embed_dim"],
        )

        features = net(
            vectors, senders, receivers, species, mask)

        if mode == "energy":
            per_atom_energies, = features.array.T
            per_atom_energies = AtomicEnergyLayer(n_species)(
                per_atom_energies, species)
            per_atom_energies *= mask

            if per_particle:
                return per_atom_energies
            else:
                return md_util.high_precision_sum(per_atom_energies)
            
        elif mode == "energy_and_shift":
            shift = MACE_shift(unique_species=mace_kwargs["unique_species"],
                               species_indep_shift=mace_kwargs["species_indep_shift"])

            per_atom_energies, = features.array.T
            per_atom_energies = AtomicEnergyLayer(n_species)(per_atom_energies, species)
            per_atom_energies *= mask

            if per_particle:
                return per_atom_energies
            else:
                # MACE potential predictions
                energy = md_util.high_precision_sum(per_atom_energies)
                # per species potential shift
                dU = shift(species)
                return energy + dU

        elif mode == "energy_and_charge":
            print(f"Shape of the outputs is {features.array.shape}")

            # Learnable scale and shift
            charge_scale = hk.get_parameter("charge_scale", shape=(), init=hk.initializers.Constant(1.0))

            per_atom_energies, per_node_charges = features.array.T
            per_atom_energies = AtomicEnergyLayer(n_species)(features.array, species)
            per_atom_energies *= mask

            # Enforce charge neutrality
            per_node_charges = charge_scale * per_node_charges
            per_node_charges -= jnp.sum(mask * per_node_charges) / jnp.sum(mask)
            per_node_charges *= mask

            if per_particle:
                return per_atom_energies, per_node_charges
            else:
                total_energy = md_util.high_precision_sum(per_atom_energies)
                return total_energy, per_node_charges

        elif mode == "property_prediction":
            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)


def mace_consol_neighborlist_pp(displacement: space.DisplacementFn,
                         r_cutoff: float,
                         n_species: int = 100,
                         n_solv_species: int = 1,
                         positions_test: jnp.ndarray = None,
                         neighbor_test: partition.NeighborList = None,
                         max_edge_multiplier: float = 1.25,
                         max_edges=None,
                         avg_num_neighbors: float = None,
                         mode: str = "energy",
                         solvent_interaction: str = "input",
                         per_particle: bool = False,
                         positive_species: bool = False,
                         **mace_kwargs
                         ) -> Tuple[nn.InitFn, Callable[[Any, md_util.Array],
                                                               md_util.Array]]:
    """NequIP model for property prediction.

    Args:
        displacement: Jax_md displacement function
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list
        n_species: Number of different atom species the network is supposed
            to process.
        positions_test: Sample positions to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        neighbor_test: Sample neighborlist to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        max_edge_multiplier: Multiplier for initial estimate of maximum edges.
        max_edges: Expected maximum of valid edges.
        nequip_escn: Use NequIPESCN instead of NequIP (more computational
            efficient).
        avg_num_neighbors: Average number of neighbors per particle. Guessed
            if positions_test and neighbor_test are provided.
        mode: Prediction mode of the model. If "property_prediction" (default),
            returns the learned node features. If "energy_prediction", returns
            the total energy of the system.
        per_particle: Return per-particle energies instead of total energy.
        positive_species: True if the smallest occurring species is 1, e.g., in
            case of atomic numbers.
        mace_kwargs: Kwargs to change the default structure of NequIP.
            For definition of the kwargs, see NequIP.


    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters. The energy function requires the same input as
        other energy functions with neighbor lists in jax_md.energy.
    """
    kwargs = deepcopy(mace_default_kwargs)
    kwargs.update(mace_kwargs)

    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    # Checking only necessary if neighbor list is dense
    _avg_num_neighbors = None
    if positions_test is not None and neighbor_test is not None:
        if neighbor_test.format == partition.Dense:
            print('Capping edges and triplets. Beware of overflow, which is'
                  ' currently not being detected.')

            testgraph, _ = sparse_graph.sparse_graph_from_neighborlist(
                displacement, positions_test, neighbor_test, r_cutoff)
            max_edges = jnp.int32(jnp.ceil(testgraph.n_edges * max_edge_multiplier))

            # cap maximum edges and angles to avoid overflow from multiplier
            n_particles, n_neighbors = neighbor_test.idx.shape
            max_edges = min(max_edges, n_particles * n_neighbors)

            print(f"Estimated max. {max_edges} edges.")

            _avg_num_neighbors = testgraph.n_edges / n_particles
        else:
            n_particles = neighbor_test.idx.shape[0]
            _avg_num_neighbors = onp.sum(neighbor_test.idx[0] < n_particles)
            _avg_num_neighbors /= n_particles

    if avg_num_neighbors is None:
        avg_num_neighbors = _avg_num_neighbors
    assert avg_num_neighbors is not None, (
        "Average number of neighbors not set and no test graph was provided."
    )

    @hk.without_apply_rng
    @hk.transform
    def model(position: md_util.Array,
              neighbor: partition.NeighborList,
              species: md_util.Array = None,
              solv_species: md_util.Array = None,
              mask: md_util.Array = None,
              **dynamic_kwargs):
        if species is None:
            print(f"[MACE] Use default species")
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        elif positive_species:
            species -= 1
        if mask is None:
            print(f"[MACE] Use default mask")
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        if solv_species is None:
            print(f"[MACE_CONSOL] Use default solvent species")
            solv_species = jnp.zeros(position.shape[0], dtype=jnp.int32)

        if solvent_interaction == "output":
            print(f"[MACE_CONSOL] Use output solvent embedding")
        
        if kwargs["block_name"] is not None:
            print(f"[MACE_CONSOL] Use {kwargs['block_name']} block")

        # Compute the displacements for all edges
        dyn_displacement = functools.partial(displacement, **dynamic_kwargs)

        if neighbor.format == partition.Dense:
            graph, _ = sparse_graph.sparse_graph_from_neighborlist(
                dyn_displacement, position, neighbor, r_cutoff,
                species, max_edges=max_edges, species_mask=mask
            )
            senders = graph.idx_i
            receivers = graph.idx_j
        else:
            assert neighbor.idx.shape == (2, neighbor.idx.shape[1]), "Neighbor list has wrong shape."
            senders, receivers = neighbor.idx

        invalid_idx = position.shape[0]

        # Set invalid edges to the cutoff to avoid numerical issues
        vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
        vectors = jnp.where(
            jnp.logical_and(
                senders < invalid_idx,
                receivers < position.shape[0]
            )[:, jnp.newaxis], vectors, r_cutoff / jnp.sqrt(3))
        vectors /= r_cutoff

        # Sort vectors by length and remove up to max_edges edges
        # lengths = jnp.linalg.norm(vectors, axis=-1)
        # sort_idx = jnp.argsort(lengths)
        # vectors = vectors[sort_idx][:max_edges]
        # senders = senders[sort_idx][:max_edges]
        # receivers = receivers[sort_idx][:max_edges]

        vectors = e3nn.IrrepsArray(
            e3nn.Irreps(kwargs['input_irreps']), vectors
        )

        net = MACE_CONSOL(
            output_irreps=kwargs["output_irreps"],
            num_interactions=kwargs["num_interactions"],
            hidden_irreps=kwargs["hidden_irreps"],
            readout_mlp_irreps=kwargs["readout_irreps"],
            avg_num_neighbors=avg_num_neighbors,
            num_species=n_species,
            max_ell=kwargs["max_ell"],
            n_radial_basis=kwargs["n_radial_basis"],
            envelope_p=kwargs["envelope_p"],
            correlation=kwargs["correlation"],
            embed_dim=kwargs["embed_dim"],
            embed_solvent_dim=kwargs["embed_solvent_dim"],
            num_solvent_species=n_solv_species,
            solvent_interaction=solvent_interaction,
            nonlinear_solembed=kwargs["nonlinear_solembed"],
            solv_tp = kwargs["solv_tp"],
            block_name=kwargs["block_name"],
        )

        features = net(vectors, senders, receivers, species, solv_species, mask)

        if mode == "energy":
            per_atom_energies, = features.array.T

            if solvent_interaction == "input":
                per_atom_energies = AtomicEnergyLayer(n_species)(per_atom_energies, species)
            if solvent_interaction == "output":
                per_atom_energies = AtomicEnergyLayer(n_species)(per_atom_energies, species)
                per_atom_energies = OutputSolventEmbedding(n_solv_species)(per_atom_energies, solv_species)                

            per_atom_energies *= mask

            if per_particle:
                return per_atom_energies
            else:
                return md_util.high_precision_sum(per_atom_energies)

        elif mode == "energy_and_charge":
            print(f"Shape of the outputs is {features.array.shape}")

            # Learnable scale and shift
            charge_scale = hk.get_parameter("charge_scale", shape=(), init=hk.initializers.Constant(1.0))

            per_atom_energies, per_node_charges = features.array.T
            per_atom_energies = AtomicEnergyLayer(n_species)(features.array, species)
            per_atom_energies *= mask

            # Enforce charge neutrality
            per_node_charges = charge_scale * per_node_charges
            per_node_charges -= jnp.sum(mask * per_node_charges) / jnp.sum(mask)
            per_node_charges *= mask

            if per_particle:
                return per_atom_energies, per_node_charges
            else:
                total_energy = md_util.high_precision_sum(per_atom_energies)
                return total_energy, per_node_charges

        elif mode == "property_prediction":
            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)


class MACE_CONSOL_VISUAL(hk.Module):
    def __init__(
        self,
        *,
        output_irreps: e3nn.Irreps,  # Irreps of the output, default 1x0e
        num_interactions: int,  # Number of interactions (layers), default 2
        hidden_irreps: e3nn.Irreps,  # 256x0e or 128x0e + 128x1o
        readout_mlp_irreps: e3nn.Irreps,  # Hidden irreps of the MLP in last readout, default 16x0e
        avg_num_neighbors: float,
        num_species: int,
        num_features: int = None,  # Number of features per node, default gcd of hidden_irreps multiplicities
        n_radial_basis: int = 8,  # Number of radial basis functions
        envelope_p: int = 6,  # Order of the envelope polynomial
        max_ell: int = 3,  # Max spherical harmonic degree, default 3
        epsilon: Optional[float] = None,
        correlation: int = 3,  # Correlation order at each layer (~ node_features^correlation), default 3
        activation: Callable = jax.nn.silu,  # activation function
        gate: Callable = jax.nn.sigmoid,  # gate function
        soft_normalization: Optional[float] = None,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
        interaction_irreps: Union[str, e3nn.Irreps] = "o3_restricted",  # or o3_full
        skip_connection_first_layer: bool = False,
        embed_dim: int = 32,
        embed_solvent_dim: int = 32,
        num_solvent_species: int,
        solvent_interaction: str = "input",  # or "output"
        nonlinear_solembed: bool = False,
        block_name: Optional[str] = None,
        solv_tp: bool = False
    ):
        super().__init__()

        output_irreps = e3nn.Irreps(output_irreps)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)

        if num_features is None:
            self.num_features = functools.reduce(
                math.gcd, (mul for mul, _ in hidden_irreps)
            )
            self.hidden_irreps = e3nn.Irreps(
                [(mul // self.num_features, ir) for mul, ir in hidden_irreps]
            )
        else:
            self.num_features = num_features
            self.hidden_irreps = hidden_irreps

        if interaction_irreps == "o3_restricted":
            self.interaction_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        elif interaction_irreps == "o3_full":
            self.interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(max_ell))
        else:
            self.interaction_irreps = e3nn.Irreps(interaction_irreps)

        self.correlation = correlation
        self.epsilon = epsilon
        self.readout_mlp_irreps = readout_mlp_irreps
        self.activation = activation
        self.gate = gate
        self.num_interactions = num_interactions
        self.output_irreps = output_irreps
        self.num_species = num_species
        self.num_solvent_species = num_solvent_species
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal
        self.max_ell = max_ell
        self.soft_normalization = soft_normalization
        self.skip_connection_first_layer = skip_connection_first_layer
        self.solvent_interaction = solvent_interaction
        self.solv_tp = solv_tp

        epsilon = hk.get_parameter(
            "varepsilon", shape=(),
            init=hk.initializers.Constant(jnp.sqrt(avg_num_neighbors))
        )
        self.epsilon = 1 / jnp.sqrt(1 + epsilon ** 2)

        # Embeddings
        if solvent_interaction == "input":
            self.node_embedding = LinearNodeEmbeddingVisual(
                embed_dim, self.num_species, self.num_features * self.hidden_irreps, 
                embed_solvent_dim, self.num_solvent_species, nonlinear_solembed, block_name
            )
        elif solvent_interaction == "output":
            self.node_embedding = LinearNodeEmbeddingLayer(
                embed_dim, self.num_species, self.num_features * self.hidden_irreps
            )

        self.radial_embedding = RadialBesselLayer(
            cutoff=1.0, num_radial=n_radial_basis, envelope_p=envelope_p
        )

        if solv_tp:
            self.FiLMLayer = FiLMLayer_e3nn(self.output_irreps)

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        solvent_specie: jnp.ndarray,
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
        is_training: bool = False,
    ) -> e3nn.IrrepsArray:
        assert vectors.ndim == 2 and vectors.shape[1] == 3
        assert node_species.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        if node_mask is None:
            node_mask = jnp.ones(node_species.shape[0], dtype=jnp.bool_)

        # Embeddings
        if self.solvent_interaction == "input":
            if self.solv_tp:
                node_feats, solv_feats, attn_weights = self.node_embedding(node_species, solvent_specie, solv_tp = self.solv_tp)

        rbf = self.radial_embedding(safe_norm(vectors.array, axis=-1))

        # Interactions
        outputs = []
        node_outputs = None
        for i in range(self.num_interactions):
            first = i == 0
            last = i == self.num_interactions - 1

            hidden_irreps = (
                self.hidden_irreps
                if not last
                else self.hidden_irreps.filter(self.output_irreps)
            )

            
            node_outputs, node_feats = MACELayer(
                first=first,
                last=last,
                num_features=self.num_features,
                interaction_irreps=self.interaction_irreps,
                hidden_irreps=hidden_irreps,
                max_ell=self.max_ell,
                epsilon=self.epsilon,
                activation=self.activation,
                gate=self.gate,
                num_species=self.num_species,
                correlation=self.correlation,
                output_irreps=self.output_irreps,
                readout_mlp_irreps=self.readout_mlp_irreps,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                soft_normalization=self.soft_normalization,
                skip_connection_first_layer=self.skip_connection_first_layer,
                solv_tp = self.solv_tp,
                name=f"layer_{i}",
            )(
                vectors,
                node_feats,
                node_species,
                rbf,
                senders,
                receivers,
                node_mask,
                is_training,
                solv_feats=solv_feats,
            )
            outputs += [node_outputs]

        return attn_weights

def mace_consol_visual(displacement: space.DisplacementFn,
                         r_cutoff: float,
                         n_species: int = 100,
                         n_solv_species: int = 1,
                         positions_test: jnp.ndarray = None,
                         neighbor_test: partition.NeighborList = None,
                         max_edge_multiplier: float = 1.25,
                         max_edges=None,
                         avg_num_neighbors: float = None,
                         mode: str = "energy",
                         solvent_interaction: str = "input",
                         per_particle: bool = False,
                         positive_species: bool = False,
                         **mace_kwargs
                         ) -> Tuple[nn.InitFn, Callable[[Any, md_util.Array],
                                                               md_util.Array]]:
    """NequIP model for property prediction.

    Args:
        displacement: Jax_md displacement function
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list
        n_species: Number of different atom species the network is supposed
            to process.
        positions_test: Sample positions to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        neighbor_test: Sample neighborlist to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        max_edge_multiplier: Multiplier for initial estimate of maximum edges.
        max_edges: Expected maximum of valid edges.
        nequip_escn: Use NequIPESCN instead of NequIP (more computational
            efficient).
        avg_num_neighbors: Average number of neighbors per particle. Guessed
            if positions_test and neighbor_test are provided.
        mode: Prediction mode of the model. If "property_prediction" (default),
            returns the learned node features. If "energy_prediction", returns
            the total energy of the system.
        per_particle: Return per-particle energies instead of total energy.
        positive_species: True if the smallest occurring species is 1, e.g., in
            case of atomic numbers.
        mace_kwargs: Kwargs to change the default structure of NequIP.
            For definition of the kwargs, see NequIP.


    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters. The energy function requires the same input as
        other energy functions with neighbor lists in jax_md.energy.
    """
    kwargs = deepcopy(mace_default_kwargs)
    kwargs.update(mace_kwargs)

    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    # Checking only necessary if neighbor list is dense
    _avg_num_neighbors = None
    if positions_test is not None and neighbor_test is not None:
        if neighbor_test.format == partition.Dense:
            print('Capping edges and triplets. Beware of overflow, which is'
                  ' currently not being detected.')

            testgraph, _ = sparse_graph.sparse_graph_from_neighborlist(
                displacement, positions_test, neighbor_test, r_cutoff)
            max_edges = jnp.int32(jnp.ceil(testgraph.n_edges * max_edge_multiplier))

            # cap maximum edges and angles to avoid overflow from multiplier
            n_particles, n_neighbors = neighbor_test.idx.shape
            max_edges = min(max_edges, n_particles * n_neighbors)

            print(f"Estimated max. {max_edges} edges.")

            _avg_num_neighbors = testgraph.n_edges / n_particles
        else:
            n_particles = neighbor_test.idx.shape[0]
            _avg_num_neighbors = onp.sum(neighbor_test.idx[0] < n_particles)
            _avg_num_neighbors /= n_particles

    if avg_num_neighbors is None:
        avg_num_neighbors = _avg_num_neighbors
    assert avg_num_neighbors is not None, (
        "Average number of neighbors not set and no test graph was provided."
    )

    @hk.without_apply_rng
    @hk.transform
    def model(position: md_util.Array,
              neighbor: partition.NeighborList,
              species: md_util.Array = None,
              solv_species: md_util.Array = None,
              mask: md_util.Array = None,
              **dynamic_kwargs):
        if species is None:
            print(f"[MACE] Use default species")
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        elif positive_species:
            species -= 1
        if mask is None:
            print(f"[MACE] Use default mask")
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        if solv_species is None:
            print(f"[MACE_CONSOL] Use default solvent species")
            solv_species = jnp.zeros(position.shape[0], dtype=jnp.int32)

        if solvent_interaction == "output":
            print(f"[MACE_CONSOL] Use output solvent embedding")
        
        if kwargs["block_name"] is not None:
            print(f"[MACE_CONSOL] Use {kwargs['block_name']} block")

        # Compute the displacements for all edges
        dyn_displacement = functools.partial(displacement, **dynamic_kwargs)

        if neighbor.format == partition.Dense:
            graph, _ = sparse_graph.sparse_graph_from_neighborlist(
                dyn_displacement, position, neighbor, r_cutoff,
                species, max_edges=max_edges, species_mask=mask
            )
            senders = graph.idx_i
            receivers = graph.idx_j
        else:
            assert neighbor.idx.shape == (2, neighbor.idx.shape[1]), "Neighbor list has wrong shape."
            senders, receivers = neighbor.idx

        invalid_idx = position.shape[0]

        # Set invalid edges to the cutoff to avoid numerical issues
        vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
        vectors = jnp.where(
            jnp.logical_and(
                senders < invalid_idx,
                receivers < position.shape[0]
            )[:, jnp.newaxis], vectors, r_cutoff / jnp.sqrt(3))
        vectors /= r_cutoff

        # Sort vectors by length and remove up to max_edges edges
        # lengths = jnp.linalg.norm(vectors, axis=-1)
        # sort_idx = jnp.argsort(lengths)
        # vectors = vectors[sort_idx][:max_edges]
        # senders = senders[sort_idx][:max_edges]
        # receivers = receivers[sort_idx][:max_edges]

        vectors = e3nn.IrrepsArray(
            e3nn.Irreps(kwargs['input_irreps']), vectors
        )

        net = MACE_CONSOL_VISUAL(
            output_irreps=kwargs["output_irreps"],
            num_interactions=kwargs["num_interactions"],
            hidden_irreps=kwargs["hidden_irreps"],
            readout_mlp_irreps=kwargs["readout_irreps"],
            avg_num_neighbors=avg_num_neighbors,
            num_species=n_species,
            max_ell=kwargs["max_ell"],
            n_radial_basis=kwargs["n_radial_basis"],
            envelope_p=kwargs["envelope_p"],
            correlation=kwargs["correlation"],
            embed_dim=kwargs["embed_dim"],
            embed_solvent_dim=kwargs["embed_solvent_dim"],
            num_solvent_species=n_solv_species,
            solvent_interaction=solvent_interaction,
            nonlinear_solembed=kwargs["nonlinear_solembed"],
            solv_tp = kwargs["solv_tp"],
            block_name=kwargs["block_name"],
        )

        attn_weights = net(vectors, senders, receivers, species, solv_species, mask)

        return attn_weights

    return jax.jit(model.init), jax.jit(model.apply)
