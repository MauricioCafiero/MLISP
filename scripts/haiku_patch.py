"""Give hk.MultiHeadAttention the `return_attn_weights` kwarg ConSolv calls.

ConSolv (chemutils/models/mace/test_block.py) calls
hk.MultiHeadAttention(...)(q, k, v, return_attn_weights=...), which no released
dm-haiku supports -- the authors evidently ran a patched haiku. This is
haiku 0.0.12's __call__ verbatim (same submodule names, so published params
load), returning (out, attn_weights) when asked. Import before building a model.
"""
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _call(self, query, key, value, mask=None, return_attn_weights=False):
    *leading_dims, sequence_length, _ = query.shape
    projection = self._linear_projection
    query_heads = projection(query, self.key_size, "query")
    key_heads = projection(key, self.key_size, "key")
    value_heads = projection(value, self.value_size, "value")
    attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    attn_logits = attn_logits / np.sqrt(self.key_size).astype(key.dtype)
    if mask is not None:
        attn_logits = jnp.where(mask, attn_logits, -1e30)
    attn_weights = jax.nn.softmax(attn_logits)
    attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))
    out = hk.Linear(self.model_size, w_init=self.w_init,
                    with_bias=self.with_bias, b_init=self.b_init)(attn)
    return (out, attn_weights) if return_attn_weights else out


assert hk.__version__ == "0.0.12", f"patch written against haiku 0.0.12, got {hk.__version__}"
hk.MultiHeadAttention.__call__ = _call
