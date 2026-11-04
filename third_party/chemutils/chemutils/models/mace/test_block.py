import functools
from pathlib import Path
from typing import Callable, Optional, Tuple, Set, Union

import e3nn_jax as e3nn
import haiku as hk
import jax.nn as jax_nn
import jax.numpy as jnp
import jax

import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_DESCRIPTOR_PATH = _REPOSITORY_ROOT / "upstream" / "descriptor.csv"  # consolv: was dataset/descriptor.csv

def descriptor_init():
    descriptor = pd.read_csv(_DESCRIPTOR_PATH)
    zero_row_series = pd.Series(0, index=descriptor.columns, name='zero_row')
    zero_row_df = zero_row_series.to_frame().T 
    descriptor = pd.concat([zero_row_df, descriptor], ignore_index=True)
    descriptor = jnp.asarray(descriptor.iloc[:, 1:])

    # normalization
    descriptor_mean = jnp.mean(descriptor, axis=0)
    descriptor_std = jnp.std(descriptor, axis=0)
    descriptor = (descriptor - descriptor_mean) / descriptor_std
    return descriptor

def positional_transform_function(num_rows: int, num_cols: int, l_0: float = 148.0) -> jnp.ndarray:
    I = jnp.arange(num_rows)[:, jnp.newaxis] 
    n = jnp.arange(num_cols)[jnp.newaxis, :] 
    
    is_even = (n % 2 == 0)
    
    even_angles = (1.0 + n/2.0) * jnp.pi * I / l_0
    
    odd_angles = (1.0 + (n-1.0)/2.0) * jnp.pi * I / l_0
    
    result = jnp.where(
        is_even,
        jnp.cos(even_angles),  
        jnp.sin(odd_angles)   
    )
    
    return result

def positional_transform(num_solvent_species: int, embed_solvent_dim: int) -> jnp.ndarray:
    matrix = positional_transform_function(num_solvent_species, embed_solvent_dim)
    return matrix
    

class PositionalEmbeddingBlock(hk.Module):
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = None,
                 gate: Optional[Callable] = None):
        super().__init__()
        self.num_solvent_species = num_solvent_species
        self.embedding_sol = positional_transform(num_solvent_species, embed_solvent_dim)
        self.irreps_out = irreps_out
        
        self.FiLM = FiLMLayer(self.irreps_out.dim,
                             activation=activation)
        
        self.activation = activation if activation else jax_nn.tanh
    
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        embedding_sol = jnp.take(self.embedding_sol, solvent_specie, axis=0)
        embedding = jnp.concatenate([embedding_sol, embedding_atom], axis=-1)
        
        gamma, beta = self.FiLM(embedding)
        
        if len(embedding_atom.shape) > len(gamma.shape):
            gamma = jnp.expand_dims(gamma, axis=1)
            beta = jnp.expand_dims(beta, axis=1)
            
        modulated = embedding_atom * (gamma) + beta
        
        if solv_tp:
            return modulated, embedding
        else:
            return modulated
    

class NonLinearSolventEmbeddingBlock(hk.Module):
    '''Non linear solvent embedding, normal initialization for embedding and weights, 1 layer, output scale'''
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = None,
                 gate: Optional[Callable] = None):
        super().__init__()
        self.num_solvent_species = num_solvent_species
        self.embedding_sol = hk.Embed(num_solvent_species, 
                                      embed_solvent_dim,
                                      embedding_matrix=None,
                                      w_init=None,)
        self.irreps_out = irreps_out
        self.mlp = e3nn.haiku.MultiLayerPerceptron(
            [self.irreps_out.dim],
            with_bias=True,
            act=activation,
            output_activation=False,
        )
        self.gate = gate
        self.output_scale = hk.get_parameter(
                                "embedding_output_scale", 
                                shape=[1], 
                                dtype=jnp.float32,
                                init=hk.initializers.Constant(1e-3) 
                            )
    
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp = False) -> e3nn.IrrepsArray:
        embedding_sol = self.embedding_sol(solvent_specie)
        embedding = jnp.concatenate([embedding_sol, embedding_atom], axis=-1)
        embedding = self.mlp(embedding)
        if solv_tp:
            return embedding * self.output_scale, embedding_sol
        else:
            return embedding * self.output_scale
    
class NonLinearSolventEmbeddingBlock1(hk.Module):
    '''Non linear solvent embedding, normal initialization for embedding and small initialization for weights, 1 layer, no output scale'''
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = None,
                 ):
        super().__init__()
        self.num_solvent_species = num_solvent_species
        self.embedding_sol = hk.Embed(num_solvent_species, 
                                      embed_solvent_dim,
                                      embedding_matrix=None,
                                      w_init=None,)
        self.irreps_out = irreps_out
        mlp = hk.Sequential([
            hk.Linear(self.irreps_out.dim, with_bias=False, w_init = hk.initializers.TruncatedNormal(stddev=1e-3, upper=0.1, lower=-0.1)),
            activation,
        ])
        self.mlp = mlp
    
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        embedding_sol = self.embedding_sol(solvent_specie)
        embedding_modulation = jnp.concatenate([embedding_sol, embedding_atom], axis=-1)
        embedding = self.mlp(embedding_modulation)
        if solv_tp:
            return embedding, embedding_sol
        else:
            return embedding
    
class NonLinearSolventEmbeddingBlock1_1(hk.Module):
    '''Non linear solvent embedding, normal initialization for embedding and small initialization for weights, multuple layers, no output scale'''
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = jax.nn.tanh,
                 gate: Optional[Callable] = None):
        super().__init__()
        self.num_solvent_species = num_solvent_species
        self.embedding_sol = hk.Embed(num_solvent_species, 
                                      embed_solvent_dim,
                                      embedding_matrix=None,
                                      w_init=None,)
        self.irreps_out = irreps_out
        self.init_std = 1e-1
        mlp = hk.Sequential([
            hk.Linear(self.irreps_out.dim, with_bias=False, w_init = hk.initializers.TruncatedNormal(stddev=self.init_std, upper=0.1, lower=-0.1)),
            activation,
            hk.Linear(self.irreps_out.dim, with_bias=False, w_init = hk.initializers.TruncatedNormal(stddev=self.init_std, upper=0.1, lower=-0.1)),
            activation,
            hk.Linear(self.irreps_out.dim, with_bias=False, w_init = hk.initializers.TruncatedNormal(stddev=self.init_std, upper=0.1, lower=-0.1)),
        ])
        self.mlp = mlp
    
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        embedding_sol = self.embedding_sol(solvent_specie)
        embedding_sol = jnp.concatenate([embedding_sol, embedding_atom], axis=-1)
        embedding = self.mlp(embedding_sol)
        if solv_tp:
            return embedding, embedding_sol
        else:
            return embedding
    
class FiLMLayer(hk.Module):
    def __init__(self, feature_dim: int, 
                 *,
                 activation: Optional[Callable] = jax.nn.tanh,
                 ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.gamma_beta = hk.Sequential([
            hk.Linear(feature_dim * 2, with_bias=False, w_init = hk.initializers.TruncatedNormal(stddev=1e-1, upper=0.1, lower=-0.1)),
            activation,
        ])
    
    def __call__(self, embedding_solvent) -> e3nn.IrrepsArray:
        gamma_beta = self.gamma_beta(embedding_solvent)
        gamma = gamma_beta[..., :self.feature_dim]  
        beta = gamma_beta[..., self.feature_dim:] 

        return gamma, beta

class FiLMLayer_e3nn(hk.Module):
    def __init__(self, feature_dim: e3nn.Irreps, 
                 *,
                 activation: Optional[Callable] = jax.nn.tanh,
                 ):
        super().__init__()
        
        self.irreps_out = feature_dim
        self.feature_dim = feature_dim.dim
        self.gamma_beta = hk.Sequential([
            hk.Linear(self.feature_dim * 2, with_bias=False, w_init = hk.initializers.TruncatedNormal(stddev=1.)),
            activation,
        ])
    
    def __call__(self, embedding_solvent) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        gamma_beta = self.gamma_beta(embedding_solvent.array)
        # gamma = e3nn.IrrepsArray(self.irreps_out, gamma_beta[..., :self.feature_dim]+1.0)  
        # beta = e3nn.IrrepsArray(self.irreps_out, gamma_beta[..., self.feature_dim:])
        gamma = gamma_beta[..., :self.feature_dim]
        beta = gamma_beta[..., self.feature_dim:]

        return gamma, beta
    
class FiLMLayer_e3nn1(hk.Module):
    def __init__(self, feature_dim: e3nn.Irreps, 
                 *,
                 activation: Optional[Callable] = jax.nn.tanh,
                 ):
        super().__init__()
        
        self.irreps_out = e3nn.Irreps(feature_dim).filter("0e").regroup()
        self.feature_dim = self.irreps_out.dim
        self.gamma_beta = hk.Sequential([
            hk.Linear(self.feature_dim * 2, with_bias=False, w_init = hk.initializers.TruncatedNormal(stddev=1.)),
            activation,
        ])
    
    def __call__(self, embedding_solvent) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        gamma_beta = self.gamma_beta(embedding_solvent.array)
        gamma = e3nn.IrrepsArray(self.irreps_out, gamma_beta[..., :self.feature_dim])  
        beta = e3nn.IrrepsArray(self.irreps_out, gamma_beta[..., self.feature_dim:])

        return gamma, beta
    
class NonLinearSolventEmbeddingBlock2(hk.Module):
    '''Non linear solvent embedding using FiLM (Feature-wise Linear Modulation)'''
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = None,
                 ):
        super().__init__()
        self.num_solvent_species = num_solvent_species
        self.embedding_sol = hk.Embed(num_solvent_species, 
                                      embed_solvent_dim,
                                      embedding_matrix=None,
                                      w_init=None)
        
        self.irreps_out = irreps_out
        
        self.FiLM = FiLMLayer(self.irreps_out.dim,
                             activation=activation)
        
        self.activation = activation if activation else jax_nn.tanh
    
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        embedding = self.embedding_sol(solvent_specie)
        # embedding = jnp.concatenate([embedding_sol, embedding_atom], axis=-1)
        
        gamma, beta = self.FiLM(embedding)
        
        if len(embedding_atom.shape) > len(gamma.shape):
            gamma = jnp.expand_dims(gamma, axis=1)
            beta = jnp.expand_dims(beta, axis=1)
        
        modulated = embedding_atom * (gamma) + beta
        
        if solv_tp:
            return modulated, embedding
        else:
            return modulated

class Attention(hk.Module):
    def __init__(self, irreps_out: e3nn.Irreps, heads=8, w_init=None):
        super().__init__()
        """
        irreps_out: define the key size of attention,
        heads: number of parallel attention layers, default 8,
        w_init: weights initialization,
        """
        self.attention = hk.MultiHeadAttention(num_heads=heads,
                                               key_size=irreps_out.dim,
                                               w_init=w_init or hk.initializers.VarianceScaling(0.1),
                                               model_size=irreps_out.dim)

    def __call__(self, atom_embedding, solvent_embedding, return_attn_weights=False):
        out = self.attention(atom_embedding, solvent_embedding, solvent_embedding, return_attn_weights=return_attn_weights)

        return out

class NonLinearSolventEmbeddingBlock3(hk.Module):
    '''Non linear solvent embedding using cross attention'''
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = None,
                 ):
        super().__init__()
        self.num_solvent_species = num_solvent_species
        self.embedding_sol = hk.Embed(num_solvent_species, 
                                      embed_solvent_dim,
                                      embedding_matrix=None,
                                      w_init=None)
        
        self.irreps_out = irreps_out
        
        self.attention = Attention(self.irreps_out)
            
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        embedding = self.embedding_sol(solvent_specie)
        # attention_out = self.attention(embedding_atom, embedding[0:1, :])
        attention_out = self.attention(embedding_atom, embedding)

        modulated = attention_out
        
        if solv_tp:
            return attention_out, embedding
        else:
            return modulated
        
class NonLinearSolventEmbeddingBlock5(hk.Module):
    '''Non linear solvent embedding using cross attention'''
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = None,
                 ):
        super().__init__()
        self.num_solvent_species = num_solvent_species
        self.embedding_sol = hk.Embed(num_solvent_species, 
                                      embed_solvent_dim,
                                      embedding_matrix=None,
                                      w_init=None)
        
        self.irreps_out = irreps_out
        
        self.attention = Attention(self.irreps_out)
            
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        embedding = self.embedding_sol(solvent_specie)
        attention_out = self.attention(embedding_atom, embedding[0:1, :])
        # attention_out = self.attention(embedding_atom, embedding)

        modulated = attention_out
        
        if solv_tp:
            return attention_out, embedding[0:1, :]
        else:
            return modulated

class NonLinearSolventEmbeddingBlockTrainable(hk.Module):
    '''Non linear solvent embedding using cross attention'''
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = None,
                 ):
        super().__init__()
        self.num_solvent_species = num_solvent_species
        self.embedding_sol = hk.Embed(num_solvent_species, 
                                      embed_solvent_dim,
                                      embedding_matrix=None,
                                      w_init=None)
        
        self.irreps_out = irreps_out
        
        self.projection = hk.Linear(self.irreps_out.dim)
        self.attention = Attention(self.irreps_out, w_init=hk.initializers.VarianceScaling(0.1))
            
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        embedding = self.embedding_sol(solvent_specie)
        embedding_expanded = self.projection(embedding[0, ...][..., jnp.newaxis])
        attention_out = self.attention(embedding_atom, embedding_expanded)
        # attention_out = self.attention(embedding_atom, embedding)

        modulated = attention_out
        
        if solv_tp:
            return attention_out, embedding[0:1, :]
        else:
            return modulated
        
class NonLinearSolventEmbeddingBlock6(hk.Module):
    '''same implementation as descriptor attention block, but using trainable solvent embedding'''
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = None,
                 ):
        super().__init__()
        self.num_solvent_species = num_solvent_species 
        self.embedding_sol = hk.Embed(num_solvent_species, 
                                embed_solvent_dim,
                                embedding_matrix=None,
                                w_init=None)  
        self.irreps_out = irreps_out
        
        self.projection = hk.Linear(self.irreps_out.dim)
        self.attention = Attention(self.irreps_out, w_init=hk.initializers.VarianceScaling(0.1))
            
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        solvent_embedding = self.embedding_sol(solvent_specie)
        solvent_embedding_vac = jnp.take(solvent_embedding, 0, axis=0)
        projected_embedding = self.projection(solvent_embedding_vac[..., jnp.newaxis])

        attention_out = self.attention(embedding_atom, projected_embedding)
            
        modulated = attention_out

        if solv_tp:
            return modulated, solvent_embedding_vac[jnp.newaxis, ...]
        else:
            return modulated
        
class NonLinearSolventEmbeddingBlock4(hk.Module):
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 heads: int = 8,
                 activation: Optional[Callable] = None,
                 ):
        super().__init__()
        self.num_solvent_species = num_solvent_species   
        self.irreps_out = irreps_out
        
        self.projection = hk.Linear(self.irreps_out.dim)
        self.attention = Attention(self.irreps_out, w_init=hk.initializers.VarianceScaling(0.1), heads=heads)

        self.descriptor = descriptor_init()
            
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        descriptors = jnp.take(self.descriptor, solvent_specie[0], axis=0)

        descriptors_expanded = descriptors[..., jnp.newaxis]

        projected_descriptors = self.projection(descriptors_expanded)

        attention_out = self.attention(embedding_atom, projected_descriptors)
            
        modulated = attention_out

        if solv_tp:
            return attention_out, descriptors[jnp.newaxis, ...]
        else:
            return modulated
        
class NonLinearSolventEmbeddingBlockDescriptor(hk.Module):
    '''Non linear solvent embedding using cross attention'''
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = None,
                 ):
        super().__init__()
        self.num_solvent_species = num_solvent_species   
        self.irreps_out = irreps_out
        
        self.attention = Attention(self.irreps_out, w_init=hk.initializers.VarianceScaling(0.1))

        self.descriptor = descriptor_init()
            
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        descriptors = jnp.take(self.descriptor, solvent_specie[0], axis=0)

        descriptors_expanded = descriptors[..., jnp.newaxis]

        attention_out = self.attention(embedding_atom, descriptors_expanded)
            
        modulated = attention_out

        if solv_tp:
            return attention_out, descriptors[jnp.newaxis, ...]
        else:
            return modulated
        
class NonLinearSolventEmbeddingBlock_film_descriptor(hk.Module):
    '''Non linear solvent embedding using FiLM'''
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = None,
                 ):
        super().__init__()
        self.num_solvent_species = num_solvent_species
        self.descriptor = descriptor_init()
        
        self.irreps_out = irreps_out
        
        self.FiLM = FiLMLayer(self.irreps_out.dim,
                             activation=activation)
        
        self.activation = activation if activation else jax_nn.tanh
    
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        descriptors = jnp.take(self.descriptor, solvent_specie, axis=0)
        # embedding = jnp.concatenate([embedding_sol, embedding_atom], axis=-1)

        gamma, beta = self.FiLM(descriptors)

        if len(embedding_atom.shape) > len(gamma.shape):
            gamma = jnp.expand_dims(gamma, axis=1)
            beta = jnp.expand_dims(beta, axis=1)
        
        modulated = embedding_atom * (gamma) + beta
        
        if solv_tp:
            return modulated, descriptors
        else:
            return modulated

class NonLinearSolventEmbeddingBlockDescriptorConcat(hk.Module):
    '''Non linear solvent embedding with descriptor concatenated, small initialization for weights, 1 layer, no output scale'''
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = None,
                 ):
        super().__init__()
        self.num_solvent_species = num_solvent_species
        self.descriptor = descriptor_init()
        self.irreps_out = irreps_out
        mlp = hk.Sequential([
            hk.Linear(self.irreps_out.dim, with_bias=False, w_init = hk.initializers.TruncatedNormal(stddev=1e-3, upper=0.1, lower=-0.1)),
            activation,
        ])
        self.mlp = mlp
    
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        descriptors = jnp.take(self.descriptor, solvent_specie, axis=0)
        embedding_modulation = jnp.concatenate([descriptors, embedding_atom], axis=-1)
        embedding = self.mlp(embedding_modulation)
        descriptors = jnp.take(self.descriptor, solvent_specie[0], axis=0)
        is_vacuum = (solvent_specie[0] == 0)
        embedding = jnp.where(is_vacuum, jnp.zeros_like(embedding), embedding)
        if solv_tp:
            return embedding, descriptors[jnp.newaxis, ...]
        else:
            return embedding

class VisualizationBlock(hk.Module):
    def __init__(self, embed_solvent_dim: int, 
                 num_solvent_species: int,
                 irreps_out: e3nn.Irreps,
                 *,
                 activation: Optional[Callable] = None,
                 ):
        super().__init__()
        self.num_solvent_species = num_solvent_species   
        self.irreps_out = irreps_out
        
        self.projection = hk.Linear(self.irreps_out.dim)
        self.attention = Attention(self.irreps_out, w_init=hk.initializers.VarianceScaling(0.1))

        self.descriptor = descriptor_init()
            
    def __call__(self, solvent_specie: jnp.ndarray, embedding_atom: jnp.ndarray, solv_tp=False) -> e3nn.IrrepsArray:
        descriptors = jnp.take(self.descriptor, solvent_specie[0], axis=0)

        descriptors_expanded = descriptors[..., jnp.newaxis]

        projected_descriptors = self.projection(descriptors_expanded)

        attention_out, attn_weights = self.attention(embedding_atom, projected_descriptors, return_attn_weights=True)
            
        modulated = attention_out

        if solv_tp:
            return attention_out, descriptors[jnp.newaxis, ...], attn_weights
        else:
            return modulated
        

# TODO: whether use activation fucntion for the direct output
block_registry = {
    "out_scale": NonLinearSolventEmbeddingBlock,
    "minor_init": NonLinearSolventEmbeddingBlock1,
    "minor_init_mlp": NonLinearSolventEmbeddingBlock1_1,
    "film": NonLinearSolventEmbeddingBlock2,
    "positional": PositionalEmbeddingBlock,
    "attention": NonLinearSolventEmbeddingBlock3,
    "descriptor": NonLinearSolventEmbeddingBlock4,
    "attention2": NonLinearSolventEmbeddingBlock5,
    "attention3": NonLinearSolventEmbeddingBlock6,
    "AttentionVisualization": VisualizationBlock,
    "descriptor_film": NonLinearSolventEmbeddingBlock_film_descriptor,
    "original_descriptor": NonLinearSolventEmbeddingBlockDescriptor,
    "trainable_attention": NonLinearSolventEmbeddingBlockTrainable,
    "descriptor_concat": NonLinearSolventEmbeddingBlockDescriptorConcat,
}

class OutputSolventEmbedding(hk.Module):
    """Adds atomic energy to the model."""
    def __init__(self, num_solvent_species: int,
            *,
            activation: Optional[Callable] = None,
            ):
        super().__init__()
        self.num_solvent_species = num_solvent_species   
        self.irreps_out = e3nn.Irreps("64x0e").filter("0e").regroup()
        
        self.projection = hk.Linear(self.irreps_out.dim)
        self.attention = Attention(self.irreps_out, w_init=hk.initializers.VarianceScaling(0.01))

        self.descriptor = descriptor_init()
            
    def __call__(self, per_atom_energy: jnp.ndarray, solvent_specie: jnp.ndarray) -> e3nn.IrrepsArray:
        descriptors = jnp.take(self.descriptor, solvent_specie[0], axis=0)

        descriptors_expanded = descriptors[..., jnp.newaxis]

        projected_descriptors = self.projection(descriptors_expanded)

        attention_out = self.attention(per_atom_energy[..., jnp.newaxis], projected_descriptors)
            
        modulated = attention_out[:, 0]

        return modulated + per_atom_energy
