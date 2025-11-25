"""
M-Theory Neural Network Module
==============================

Neural architecture with 11-dimensional latent space:
- 4D base (spacetime-like, coarse semantics)
- 7D fiber (G2 manifold, fine-grained)

Inspired by M-theory compactification on G2 holonomy manifolds.
"""

from .m_theory_network import (
    MTheoryConfig,
    MTheoryNetwork,
    MTheoryLayer,
    G2Geometry,
    FiberBundle,
    visualize_training,
    create_multiscale_task,
)

__all__ = [
    'MTheoryConfig',
    'MTheoryNetwork', 
    'MTheoryLayer',
    'G2Geometry',
    'FiberBundle',
    'visualize_training',
    'create_multiscale_task',
]

__version__ = '0.1.0'
__author__ = 'Claude (under duress)'
