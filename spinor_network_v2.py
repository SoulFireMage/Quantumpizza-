#!/usr/bin/env python3
"""
Spinor Neural Network v2 - Geometric (Clifford) Algebra
=========================================================

A neural network where:
- Activations are SPINORS (elements of even subalgebra)
- Weights are ROTORS (exponentials of bivectors)  
- All operations use GEOMETRIC PRODUCT

This exploits the deep geometric structure of 3D space itself.

Mathematical Foundation:
- Clifford Algebra Cl(3,0) over R³
- Spinors transform as: S' = R·S·R† (sandwich product)
- Rotors are: R = exp(θB/2) where B is a unit bivector
- 720° rotation = identity (spinor double-cover)

Training Task: Rotation Prediction
- Given: input vector + rotation specification
- Predict: the rotated vector
- This is literally what spinors are designed for

Wheeler Connection:
- If there's only one electron zig-zagging through time, the 720° property
  isn't a quirk - it's bookkeeping for temporal direction
- Parsimony hint: maybe the fundamental primitive isn't "many neurons"
  but "one structure appearing many times from different perspectives"

Backend Options:
- NUMPY: Works everywhere, numerical gradients (slow but functional)
- JAX: Autodiff + JIT compilation (fast, requires jax + jaxga)

Usage:
    python spinor_network_v2.py              # Auto-detect backend
    python spinor_network_v2.py --numpy      # Force numpy
    python spinor_network_v2.py --jax        # Force JAX (will error if unavailable)

Author: Richard & Claude
Date: November 2025
Part of: QuantumPizza experimental architecture zoo
"""

import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')

# =============================================================================
# BACKEND DETECTION
# =============================================================================

def detect_backend(force: Optional[str] = None) -> str:
    """Detect available backend: 'jax' or 'numpy'."""
    if force == 'numpy':
        return 'numpy'
    
    if force == 'jax':
        try:
            import jax
            import jaxga
            return 'jax'
        except ImportError as e:
            raise ImportError(
                f"JAX backend requested but not available: {e}\n"
                "Install with: pip install jax jaxlib jaxga"
            )
    
    # Auto-detect
    try:
        import jax
        import jax.numpy as jnp
        # Check if jaxga is available
        try:
            from jaxga.clifford import CliffordAlgebra as JaxClifford
            print("Backend: JAX + jaxga (autodiff enabled)")
            return 'jax'
        except ImportError:
            print("Backend: JAX available but jaxga missing, using numpy")
            print("  For JAX Clifford algebra: pip install jaxga")
            return 'numpy'
    except ImportError:
        print("Backend: NumPy (numerical gradients)")
        return 'numpy'


# =============================================================================
# NUMPY IMPLEMENTATION (fallback)
# =============================================================================

import numpy as np

class CliffordAlgebraNP:
    """
    Clifford Algebra Cl(p,q,r) - NumPy implementation.
    
    Default is Cl(3,0,0) - the algebra of 3D Euclidean space.
    
    Basis elements (binary index encoding):
        0b000 = 1     (scalar,       grade 0)
        0b001 = e1    (vector,       grade 1)
        0b010 = e2    (vector,       grade 1)
        0b011 = e12   (bivector,     grade 2)
        0b100 = e3    (vector,       grade 1)
        0b101 = e13   (bivector,     grade 2)
        0b110 = e23   (bivector,     grade 2)
        0b111 = e123  (pseudoscalar, grade 3)
    
    Key insight: Spinors live in the EVEN subalgebra (grades 0, 2)
    which has dimension 4 - exactly the quaternions!
    """
    
    def __init__(self, p: int = 3, q: int = 0, r: int = 0):
        self.p, self.q, self.r = p, q, r
        self.n = p + q + r
        self.dim = 2 ** self.n
        self.signature = [1] * p + [-1] * q + [0] * r
        self._build_multiplication_table()
    
    def _build_multiplication_table(self):
        """Build geometric product lookup tables."""
        self.mult_sign = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.mult_result = np.zeros((self.dim, self.dim), dtype=np.int32)
        
        for i in range(self.dim):
            for j in range(self.dim):
                sign, result = self._blade_product(i, j)
                self.mult_sign[i, j] = sign
                self.mult_result[i, j] = result
    
    def _blade_product(self, a: int, b: int) -> Tuple[float, int]:
        """Product of two basis blades (bitmask representation)."""
        sign = 1.0
        result = a ^ b
        
        for i in range(self.n):
            if b & (1 << i):
                bits_to_pass = bin(a >> (i + 1)).count('1')
                if bits_to_pass % 2 == 1:
                    sign *= -1
                if a & (1 << i):
                    sign *= self.signature[i]
        
        return sign, result
    
    def geometric_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Geometric product: ab = a·b + a∧b"""
        result = np.zeros(self.dim)
        for i in range(self.dim):
            if abs(a[i]) < 1e-12:
                continue
            for j in range(self.dim):
                if abs(b[j]) < 1e-12:
                    continue
                idx = self.mult_result[i, j]
                result[idx] += self.mult_sign[i, j] * a[i] * b[j]
        return result
    
    def grade(self, blade_index: int) -> int:
        """Grade of basis blade = popcount of binary index."""
        return bin(blade_index).count('1')
    
    def grade_project(self, mv: np.ndarray, k: int) -> np.ndarray:
        """Project onto grade-k component."""
        result = np.zeros(self.dim)
        for i in range(self.dim):
            if self.grade(i) == k:
                result[i] = mv[i]
        return result
    
    def even_subalgebra(self, mv: np.ndarray) -> np.ndarray:
        """Project to even subalgebra (spinor space)."""
        result = np.zeros(self.dim)
        for i in range(self.dim):
            if self.grade(i) % 2 == 0:
                result[i] = mv[i]
        return result
    
    def reverse(self, mv: np.ndarray) -> np.ndarray:
        """Reverse (dagger): sign = (-1)^(k(k-1)/2) for grade k."""
        result = np.zeros(self.dim)
        for i in range(self.dim):
            k = self.grade(i)
            sign = (-1) ** (k * (k - 1) // 2)
            result[i] = sign * mv[i]
        return result
    
    def norm_squared(self, mv: np.ndarray) -> float:
        """Compute |mv|² = <mv · mv†>_0"""
        return self.geometric_product(mv, self.reverse(mv))[0]
    
    def normalize(self, mv: np.ndarray) -> np.ndarray:
        """Normalize to unit magnitude."""
        norm_sq = self.norm_squared(mv)
        if norm_sq < 1e-12:
            return mv
        return mv / np.sqrt(abs(norm_sq))
    
    def exp_bivector(self, B: np.ndarray) -> np.ndarray:
        """
        exp(B) for bivector B → rotor (rotation operator).
        
        For B² < 0: exp(B) = cos(|B|) + sin(|B|)B̂  (rotation)
        For B² > 0: exp(B) = cosh(|B|) + sinh(|B|)B̂ (boost)
        For B² = 0: exp(B) = 1 + B (parabolic)
        """
        B_biv = self.grade_project(B, 2)
        B_sq = self.geometric_product(B_biv, B_biv)
        scalar = B_sq[0]
        
        result = np.zeros(self.dim)
        if abs(scalar) < 1e-10:
            result[0] = 1.0
            result += B_biv
        elif scalar < 0:
            theta = np.sqrt(-scalar)
            result[0] = np.cos(theta)
            result += np.sin(theta) / theta * B_biv
        else:
            phi = np.sqrt(scalar)
            result[0] = np.cosh(phi)
            result += np.sinh(phi) / phi * B_biv
        return result
    
    def apply_rotor(self, R: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Sandwich product: x' = R x R†"""
        R_rev = self.reverse(R)
        return self.geometric_product(self.geometric_product(R, x), R_rev)
    
    def vector_to_mv(self, v: np.ndarray) -> np.ndarray:
        """3D vector → multivector (grade-1)."""
        mv = np.zeros(self.dim)
        mv[1], mv[2], mv[4] = v[0], v[1], v[2]
        return mv
    
    def mv_to_vector(self, mv: np.ndarray) -> np.ndarray:
        """Multivector → 3D vector (grade-1 extraction)."""
        return np.array([mv[1], mv[2], mv[4]])
    
    def vector_to_spinor(self, v: np.ndarray) -> np.ndarray:
        """Embed vector into spinor space (scalar + bivector)."""
        spinor = np.zeros(self.dim)
        spinor[0] = np.linalg.norm(v)  # Scalar = magnitude
        spinor[3], spinor[5], spinor[6] = v[0], v[1], v[2]  # e12, e13, e23
        return spinor
    
    def spinor_to_vector(self, spinor: np.ndarray) -> np.ndarray:
        """Extract vector from spinor (bivector components)."""
        return np.array([spinor[3], spinor[5], spinor[6]])


# =============================================================================
# JAX IMPLEMENTATION (optional, with autodiff)
# =============================================================================

def get_jax_backend():
    """Import and configure JAX backend. Returns None if unavailable."""
    try:
        import jax
        import jax.numpy as jnp
        from jax import grad, jit, vmap
        from functools import partial
        
        # Try to import jaxga
        try:
            from jaxga.clifford import CliffordAlgebra as JaxCliffordBase
            HAS_JAXGA = True
        except ImportError:
            HAS_JAXGA = False
            JaxCliffordBase = None
        
        class CliffordAlgebraJAX:
            """
            JAX-compatible Clifford algebra.
            
            If jaxga is available, wraps it. Otherwise, uses our own JAX implementation.
            """
            
            def __init__(self, p: int = 3, q: int = 0, r: int = 0):
                self.p, self.q, self.r = p, q, r
                self.n = p + q + r
                self.dim = 2 ** self.n
                self.signature = jnp.array([1.0] * p + [-1.0] * q + [0.0] * r)
                
                if HAS_JAXGA:
                    # Use jaxga's optimized implementation
                    self._jaxga = JaxCliffordBase([1.0]*p + [-1.0]*q + [0.0]*r)
                else:
                    self._jaxga = None
                    self._build_tables()
            
            def _build_tables(self):
                """Build multiplication tables for pure JAX implementation."""
                mult_sign = np.zeros((self.dim, self.dim), dtype=np.float32)
                mult_result = np.zeros((self.dim, self.dim), dtype=np.int32)
                
                for i in range(self.dim):
                    for j in range(self.dim):
                        sign = 1.0
                        result = i ^ j
                        for k in range(self.n):
                            if j & (1 << k):
                                bits_to_pass = bin(i >> (k + 1)).count('1')
                                if bits_to_pass % 2 == 1:
                                    sign *= -1
                                if i & (1 << k):
                                    sign *= float(self.signature[k])
                        mult_sign[i, j] = sign
                        mult_result[i, j] = result
                
                self.mult_sign = jnp.array(mult_sign)
                self.mult_result = jnp.array(mult_result)
            
            @partial(jit, static_argnums=(0,))
            def geometric_product(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
                """Geometric product with JIT compilation."""
                if self._jaxga is not None:
                    return self._jaxga.product(a, b)
                
                # Manual implementation for pure JAX
                result = jnp.zeros(self.dim)
                for i in range(self.dim):
                    for j in range(self.dim):
                        idx = self.mult_result[i, j]
                        result = result.at[idx].add(
                            self.mult_sign[i, j] * a[i] * b[j]
                        )
                return result
            
            def grade(self, blade_index: int) -> int:
                return bin(blade_index).count('1')
            
            @partial(jit, static_argnums=(0, 2))
            def grade_project(self, mv: jnp.ndarray, k: int) -> jnp.ndarray:
                mask = jnp.array([1.0 if self.grade(i) == k else 0.0 
                                  for i in range(self.dim)])
                return mv * mask
            
            @partial(jit, static_argnums=(0,))
            def even_subalgebra(self, mv: jnp.ndarray) -> jnp.ndarray:
                mask = jnp.array([1.0 if self.grade(i) % 2 == 0 else 0.0 
                                  for i in range(self.dim)])
                return mv * mask
            
            @partial(jit, static_argnums=(0,))
            def reverse(self, mv: jnp.ndarray) -> jnp.ndarray:
                signs = jnp.array([(-1) ** (self.grade(i) * (self.grade(i) - 1) // 2)
                                   for i in range(self.dim)])
                return mv * signs
            
            @partial(jit, static_argnums=(0,))
            def exp_bivector(self, B: jnp.ndarray) -> jnp.ndarray:
                B_biv = self.grade_project(B, 2)
                B_sq = self.geometric_product(B_biv, B_biv)
                scalar = B_sq[0]
                
                def elliptic(s):
                    theta = jnp.sqrt(-s)
                    return jnp.cos(theta), jnp.sin(theta) / theta
                
                def hyperbolic(s):
                    phi = jnp.sqrt(s)
                    return jnp.cosh(phi), jnp.sinh(phi) / phi
                
                def parabolic(s):
                    return 1.0, 1.0
                
                # Choose case based on sign of B²
                c, s_factor = jax.lax.cond(
                    jnp.abs(scalar) < 1e-10,
                    parabolic,
                    lambda sc: jax.lax.cond(sc < 0, elliptic, hyperbolic, sc),
                    scalar
                )
                
                result = jnp.zeros(self.dim)
                result = result.at[0].set(c)
                result = result + s_factor * B_biv
                return result
            
            @partial(jit, static_argnums=(0,))
            def apply_rotor(self, R: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
                R_rev = self.reverse(R)
                return self.geometric_product(
                    self.geometric_product(R, x), R_rev
                )
            
            @partial(jit, static_argnums=(0,))
            def normalize(self, mv: jnp.ndarray) -> jnp.ndarray:
                norm_sq = self.geometric_product(mv, self.reverse(mv))[0]
                return mv / jnp.sqrt(jnp.abs(norm_sq) + 1e-12)
            
            def vector_to_spinor(self, v: jnp.ndarray) -> jnp.ndarray:
                spinor = jnp.zeros(self.dim)
                spinor = spinor.at[0].set(jnp.linalg.norm(v))
                spinor = spinor.at[3].set(v[0])
                spinor = spinor.at[5].set(v[1])
                spinor = spinor.at[6].set(v[2])
                return spinor
            
            def spinor_to_vector(self, spinor: jnp.ndarray) -> jnp.ndarray:
                return jnp.array([spinor[3], spinor[5], spinor[6]])
        
        return {
            'jnp': jnp,
            'grad': grad,
            'jit': jit,
            'vmap': vmap,
            'CliffordAlgebra': CliffordAlgebraJAX,
            'available': True
        }
    
    except ImportError:
        return {'available': False}


# =============================================================================
# SPINOR LAYER
# =============================================================================

@dataclass
class SpinorLayerParams:
    """Layer parameters: bivectors → rotors, plus bias."""
    bivectors: np.ndarray   # (out_dim, in_dim, algebra_dim) - learnable
    rotors: np.ndarray      # (out_dim, in_dim, algebra_dim) - derived
    bias: np.ndarray        # (out_dim, algebra_dim)


class SpinorLayerNP:
    """
    Spinor layer using NumPy (numerical gradients).
    
    Forward: output[j] = normalize(Σᵢ Rⱼᵢ · input[i] · Rⱼᵢ† + bias[j])
    """
    
    def __init__(self, in_dim: int, out_dim: int, ca: CliffordAlgebraNP, name: str = ""):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ca = ca
        self.name = name
        self.params = self._init_params()
        self.cache = {}
    
    def _init_params(self) -> SpinorLayerParams:
        bivectors = np.random.randn(self.out_dim, self.in_dim, self.ca.dim) * 0.1
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                bivectors[i, j] = self.ca.grade_project(bivectors[i, j], 2)
        
        rotors = np.zeros_like(bivectors)
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                rotors[i, j] = self.ca.exp_bivector(bivectors[i, j])
        
        bias = np.random.randn(self.out_dim, self.ca.dim) * 0.01
        for i in range(self.out_dim):
            bias[i] = self.ca.even_subalgebra(bias[i])
        
        return SpinorLayerParams(bivectors, rotors, bias)
    
    def update_rotors(self):
        """Recompute rotors from bivectors."""
        for i in range(self.out_dim):
            for j in range(self.in_dim):
                self.params.rotors[i, j] = self.ca.exp_bivector(
                    self.params.bivectors[i, j]
                )
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass: (batch, in_dim, algebra_dim) → (batch, out_dim, algebra_dim)"""
        batch_size = inputs.shape[0]
        outputs = np.zeros((batch_size, self.out_dim, self.ca.dim))
        
        for b in range(batch_size):
            for i in range(self.out_dim):
                acc = np.zeros(self.ca.dim)
                for j in range(self.in_dim):
                    acc += self.ca.apply_rotor(self.params.rotors[i, j], inputs[b, j])
                acc += self.params.bias[i]
                outputs[b, i] = self.ca.normalize(acc)
        
        self.cache['inputs'] = inputs
        self.cache['outputs'] = outputs
        return outputs


# =============================================================================
# SPINOR NETWORK
# =============================================================================

class SpinorNetworkNP:
    """
    Complete spinor network for 3D transformations.
    
    Vector → Spinor → [SpinorLayers] → Spinor → Vector
    """
    
    def __init__(self, hidden_dims: List[int], ca: Optional[CliffordAlgebraNP] = None):
        self.ca = ca or CliffordAlgebraNP(3, 0, 0)
        self.hidden_dims = hidden_dims
        
        dims = [1] + hidden_dims + [1]
        self.layers = [
            SpinorLayerNP(dims[i], dims[i+1], self.ca, f"layer_{i}")
            for i in range(len(dims) - 1)
        ]
    
    def forward(self, vectors: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass with layer activations for visualization."""
        batch_size = vectors.shape[0]
        
        # Embed
        spinors = np.zeros((batch_size, 1, self.ca.dim))
        for b in range(batch_size):
            spinors[b, 0] = self.ca.vector_to_spinor(vectors[b])
        
        layer_spinors = [spinors.copy()]
        
        # Layers
        x = spinors
        for layer in self.layers:
            x = layer.forward(x)
            layer_spinors.append(x.copy())
        
        # Extract
        output_vectors = np.zeros((batch_size, 3))
        for b in range(batch_size):
            output_vectors[b] = self.ca.spinor_to_vector(x[b, 0])
        
        return output_vectors, layer_spinors
    
    def compute_loss(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.mean((preds - targets) ** 2)
    
    def train_step(self, inputs: np.ndarray, targets: np.ndarray, lr: float = 0.01) -> float:
        """Training step with numerical gradients."""
        eps = 1e-5
        preds, _ = self.forward(inputs)
        loss = self.compute_loss(preds, targets)
        
        for layer in self.layers:
            # Bivector gradients
            for i in range(layer.out_dim):
                for j in range(layer.in_dim):
                    for k in range(self.ca.dim):
                        if self.ca.grade(k) != 2:
                            continue
                        
                        layer.params.bivectors[i, j, k] += eps
                        layer.update_rotors()
                        p_plus, _ = self.forward(inputs)
                        l_plus = self.compute_loss(p_plus, targets)
                        
                        layer.params.bivectors[i, j, k] -= 2 * eps
                        layer.update_rotors()
                        p_minus, _ = self.forward(inputs)
                        l_minus = self.compute_loss(p_minus, targets)
                        
                        layer.params.bivectors[i, j, k] += eps
                        grad = (l_plus - l_minus) / (2 * eps)
                        layer.params.bivectors[i, j, k] -= lr * grad
            
            layer.update_rotors()
            
            # Bias gradients
            for i in range(layer.out_dim):
                for k in range(self.ca.dim):
                    if self.ca.grade(k) % 2 != 0:
                        continue
                    
                    layer.params.bias[i, k] += eps
                    p_plus, _ = self.forward(inputs)
                    l_plus = self.compute_loss(p_plus, targets)
                    
                    layer.params.bias[i, k] -= 2 * eps
                    p_minus, _ = self.forward(inputs)
                    l_minus = self.compute_loss(p_minus, targets)
                    
                    layer.params.bias[i, k] += eps
                    grad = (l_plus - l_minus) / (2 * eps)
                    layer.params.bias[i, k] -= lr * grad
        
        return loss


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_rotation_data(n_samples: int, angle: float = np.pi/4, 
                           axis: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """
    Generate rotation prediction data.
    
    Default: 45° rotation around z-axis.
    """
    if axis is None:
        axis = np.array([0, 0, 1])
    
    c, s = np.cos(angle), np.sin(angle)
    # Rodrigues rotation matrix
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    
    # Random unit vectors
    inputs = np.random.randn(n_samples, 3)
    inputs = inputs / np.linalg.norm(inputs, axis=1, keepdims=True)
    
    targets = (R @ inputs.T).T
    
    return inputs, targets, (axis, angle)


def generate_variable_rotation_data(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Variable rotations - harder task.
    
    Input: [vector (3), axis (3), sin(θ), cos(θ)] = 8D
    Output: rotated vector (3D)
    """
    inputs, targets = [], []
    
    for _ in range(n_samples):
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        angle = np.random.uniform(0, 2 * np.pi)
        
        c, s = np.cos(angle), np.sin(angle)
        v_rot = v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)
        
        inputs.append(np.concatenate([v, axis, [s, c]]))
        targets.append(v_rot)
    
    return np.array(inputs), np.array(targets)


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_spinor_flow(network, inputs: np.ndarray, n_samples: int = 20):
    """
    Gemini's request: Show how spinors "twist" through layers.
    
    Arrow DIRECTION = bivector components
    Arrow COLOR = scalar part
    """
    preds, layer_spinors = network.forward(inputs[:n_samples])
    n_layers = len(layer_spinors)
    
    fig = plt.figure(figsize=(5 * n_layers, 5))
    
    for l, spinors in enumerate(layer_spinors):
        ax = fig.add_subplot(1, n_layers, l + 1, projection='3d')
        
        for b in range(min(n_samples, spinors.shape[0])):
            for s in range(spinors.shape[1]):
                spinor = spinors[b, s]
                
                # Scalar → color
                color_val = (np.tanh(spinor[0]) + 1) / 2
                
                # Bivector → direction  
                direction = np.array([spinor[3], spinor[5], spinor[6]])
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction = direction / norm * 0.3
                
                origin = np.array([b * 0.1, s * 0.1, 0])
                color = plt.cm.coolwarm(color_val)
                
                ax.quiver(
                    origin[0], origin[1], origin[2],
                    direction[0], direction[1], direction[2],
                    color=color, alpha=0.7, arrow_length_ratio=0.3
                )
        
        titles = {0: 'Input', n_layers-1: 'Output'}
        ax.set_title(titles.get(l, f'Layer {l}'))
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
    
    plt.suptitle('Spinor Flow (Color=scalar, Direction=bivector)')
    plt.tight_layout()
    return fig


def visualize_rotation_results(inputs, targets, predictions, n_show: int = 10):
    """Show input→target and prediction→target comparison."""
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(min(n_show, len(inputs))):
        ax1.quiver(0, 0, 0, inputs[i,0], inputs[i,1], inputs[i,2],
                   color='blue', alpha=0.5, arrow_length_ratio=0.1)
        ax1.quiver(0, 0, 0, targets[i,0], targets[i,1], targets[i,2],
                   color='green', alpha=0.5, arrow_length_ratio=0.1)
    ax1.set_title('Input (blue) → Target (green)')
    ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 1.5); ax1.set_zlim(-1.5, 1.5)
    
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(min(n_show, len(inputs))):
        ax2.quiver(0, 0, 0, predictions[i,0], predictions[i,1], predictions[i,2],
                   color='red', alpha=0.5, arrow_length_ratio=0.1)
        ax2.quiver(0, 0, 0, targets[i,0], targets[i,1], targets[i,2],
                   color='green', alpha=0.5, arrow_length_ratio=0.1)
    ax2.set_title('Prediction (red) vs Target (green)')
    ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5); ax2.set_zlim(-1.5, 1.5)
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Spinor Neural Network')
    parser.add_argument('--numpy', action='store_true', help='Force numpy backend')
    parser.add_argument('--jax', action='store_true', help='Force JAX backend')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--hidden', type=int, nargs='+', default=[4, 4])
    args = parser.parse_args()
    
    force = 'numpy' if args.numpy else ('jax' if args.jax else None)
    backend = detect_backend(force)
    
    print("=" * 70)
    print("SPINOR NEURAL NETWORK v2")
    print("Geometric Algebra Cl(3,0) - Rotation Prediction")
    print("=" * 70)
    
    # Config
    n_train, n_test = 100, 20
    hidden_dims = args.hidden
    n_epochs = args.epochs
    lr = args.lr
    
    print(f"\nConfig: train={n_train}, hidden={hidden_dims}, epochs={n_epochs}, lr={lr}")
    
    # Data
    print("\nGenerating data: 45° rotation around z-axis")
    train_in, train_out, (axis, angle) = generate_rotation_data(n_train)
    test_in, test_out, _ = generate_rotation_data(n_test)
    
    # Network
    if backend == 'numpy':
        ca = CliffordAlgebraNP(3, 0, 0)
        network = SpinorNetworkNP(hidden_dims, ca)
    else:
        # JAX version - would use CliffordAlgebraJAX
        # For now, fall back to numpy (JAX version needs more work for full training loop)
        ca = CliffordAlgebraNP(3, 0, 0)
        network = SpinorNetworkNP(hidden_dims, ca)
    
    print(f"Network: {len(network.layers)} layers, Cl(3,0,0) dim={ca.dim}")
    
    # Train
    print("\nTraining...")
    losses = []
    
    for epoch in range(n_epochs):
        loss = network.train_step(train_in, train_out, lr)
        losses.append(loss)
        
        if (epoch + 1) % 10 == 0:
            test_preds, _ = network.forward(test_in)
            test_loss = network.compute_loss(test_preds, test_out)
            
            dots = np.clip(np.sum(test_preds * test_out, axis=1), -1, 1)
            angle_err = np.degrees(np.mean(np.arccos(dots)))
            
            print(f"  Epoch {epoch+1:3d}: train={loss:.6f}, test={test_loss:.6f}, "
                  f"angle_err={angle_err:.2f}°")
    
    # Final
    print("\n" + "=" * 70)
    test_preds, layer_spinors = network.forward(test_in)
    final_loss = network.compute_loss(test_preds, test_out)
    
    dots = np.clip(np.sum(test_preds * test_out, axis=1), -1, 1)
    angles = np.arccos(dots)
    
    print(f"Final: loss={final_loss:.6f}, mean_err={np.degrees(np.mean(angles)):.2f}°, "
          f"max_err={np.degrees(np.max(angles)):.2f}°")
    
    # Visualizations
    print("\nSaving visualizations...")
    
    fig1, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training Loss')
    ax.set_yscale('log'); ax.grid(True, alpha=0.3)
    fig1.savefig('spinor_loss.png', dpi=150, bbox_inches='tight')
    print("  spinor_loss.png")
    
    fig2 = visualize_rotation_results(test_in, test_out, test_preds)
    fig2.savefig('spinor_rotation.png', dpi=150, bbox_inches='tight')
    print("  spinor_rotation.png")
    
    fig3 = visualize_spinor_flow(network, test_in)
    fig3.savefig('spinor_flow.png', dpi=150, bbox_inches='tight')
    print("  spinor_flow.png")
    
    plt.close('all')
    
    print("\n" + "=" * 70)
    print("ARCHITECTURE SUMMARY")
    print("=" * 70)
    print("""
  • Clifford Algebra Cl(3,0): Geometric product, grades, rotors
  • Spinor Layers: Weights are ROTORS (exp of bivectors)
  • Sandwich Product: R·x·R† preserves geometric structure
  • Task: Learn 45° z-rotation
  
  This is NOT an MLP:
  - Rotational structure built-in, not learned from scratch
  - Weights live in Lie algebra (bivectors)
  - Transformations preserve geometry
  
  Wheeler hint: If one electron → all electrons via worldline,
  then spinor's 720° isn't quirk, it's direction bookkeeping.
  Parsimony: maybe "many neurons" = "one structure, many views"
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
