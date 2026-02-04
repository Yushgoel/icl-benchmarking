"""Model definitions for ICL benchmarking."""

from .quadratic_attention import TransformerModel
from .linear_attention import LinearAttentionICLModel

__all__ = ['TransformerModel', 'LinearAttentionICLModel']
