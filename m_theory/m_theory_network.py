"""
M-Theoretic Neural Network v3
=============================

A neural architecture inspired by M-theory's dimensional structure:
- 4 "large" base dimensions (major semantic axes / spacetime)
- 7 "compactified" fiber dimensions (fine-grained / G2 manifold)

Key improvements in v3:
- Tuned compactification: fiber can breathe and learn
- Adaptive fiber radius that can expand during training
- Visualization of base vs fiber contributions
- Proper G2 holonomy with the canonical 3-form
- Ready for peer review by Kimi and Gemini (pray for me)

The Physics:
- M-theory lives in 11 dimensions
- 7 dimensions compactify on a G2 holonomy manifold
- G2 is the automorphism group of the octonions
- Ricci-flat (vacuum Einstein equations satisfied)

The ML Analogy:
- Base dimensions = coarse semantic structure
- Fiber dimensions = fine-grained distinctions
- Compactification = regularization keeping fiber "small but useful"

Repository: QuantumPizza
Author: Claude (under duress from Richard)
Date: 2024-11-24
Witnesses: Potentially Kimi K2 and Gemini (the ladies)
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import warnings


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MTheoryConfig:
    """Configuration for M-theoretic network geometry."""
    
    # Dimensional structure (must sum to 11!)
    base_dim: int = 4          # "Large" spacetime-like dimensions
    fiber_dim: int = 7         # "Compactified" G2 manifold dimensions
    total_dim: int = 11        # Full M-theory dimensionality
    
    # Compactification parameters
    fiber_radius: float = 0.3          # Initial radius (not too small!)
    min_fiber_radius: float = 0.05     # Don't crush it completely
    max_fiber_radius: float = 0.5      # Don't let it explode
    compactification_strength: float = 0.5  # Gentler regularization
    
    # Adaptive radius: let the network learn how compact to be
    adaptive_radius: bool = True
    radius_learning_rate: float = 0.001
    
    # G2 / Calabi-Yau parameters
    use_g2_holonomy: bool = True
    ricci_flat_strength: float = 0.01  # Light touch
    
    # Training parameters
    learning_rate: float = 0.01
    
    def __post_init__(self):
        assert self.base_dim + self.fiber_dim == self.total_dim, \
            f"Dimensions must sum to 11! Got {self.base_dim} + {self.fiber_dim} = {self.base_dim + self.fiber_dim}"


# ============================================================================
# G2 GEOMETRY
# ============================================================================

class G2Geometry:
    """
    Geometry of G2 holonomy manifolds for the 7D fiber.
    
    G2 is the exceptional Lie group that is the automorphism group of 
    the octonions. G2 manifolds are 7-dimensional Riemannian manifolds 
    with holonomy group contained in G2.
    
    Key properties:
    1. Ricci-flat (satisfy vacuum Einstein equations)
    2. Admit a parallel 3-form φ (the "associative calibration")
    3. Admit a parallel 4-form ψ = *φ (the "coassociative calibration")
    4. Holonomy Hol(g) ⊆ G2 ⊂ SO(7)
    
    The 3-form φ completely determines the metric and orientation.
    """
    
    def __init__(self, dim: int = 7, radius: float = 0.3):
        assert dim == 7, "G2 manifolds are 7-dimensional!"
        self.dim = dim
        self.radius = radius
        
        # Construct the canonical G2 3-form
        self.phi = self._construct_g2_3form()
        
        # The dual 4-form (Hodge star of φ)
        self.psi = self._construct_g2_4form()
        
        # Base metric (flat, scaled by radius)
        self.g = np.eye(dim) * (radius ** 2)
        
    def _construct_g2_3form(self) -> np.ndarray:
        """
        Construct the canonical G2 3-form φ on R^7.
        
        In the standard basis {e^1, ..., e^7}, the G2 3-form is:
        
        φ = e^{123} + e^{145} + e^{167} + e^{246} - e^{257} - e^{347} - e^{356}
        
        where e^{ijk} = e^i ∧ e^j ∧ e^k is a basis 3-form.
        
        This is stored as an antisymmetric (7,7,7) tensor.
        """
        phi = np.zeros((7, 7, 7))
        
        # The 7 terms of the standard G2 3-form (using 0-indexing)
        # Each tuple is (indices, sign)
        terms = [
            ((0, 1, 2), +1),   # e^{123}
            ((0, 3, 4), +1),   # e^{145}  
            ((0, 5, 6), +1),   # e^{167}
            ((1, 3, 5), +1),   # e^{246}
            ((1, 4, 6), -1),   # -e^{257}
            ((2, 3, 6), -1),   # -e^{347}
            ((2, 4, 5), -1),   # -e^{356}
        ]
        
        for indices, sign in terms:
            for perm, parity in self._antisymmetric_permutations(indices):
                phi[perm] = sign * parity
                
        return phi
    
    def _construct_g2_4form(self) -> np.ndarray:
        """
        Construct the G2 4-form ψ = *φ (Hodge dual of φ).
        
        ψ = e^{4567} + e^{2367} + e^{2345} + e^{1357} - e^{1346} - e^{1256} - e^{1247}
        
        This is the coassociative calibration.
        """
        psi = np.zeros((7, 7, 7, 7))
        
        terms = [
            ((3, 4, 5, 6), +1),   # e^{4567}
            ((1, 2, 5, 6), +1),   # e^{2367}
            ((1, 2, 3, 4), +1),   # e^{2345}
            ((0, 2, 4, 6), +1),   # e^{1357}
            ((0, 2, 3, 5), -1),   # -e^{1346}
            ((0, 1, 4, 5), -1),   # -e^{1256}
            ((0, 1, 3, 6), -1),   # -e^{1247}
        ]
        
        for indices, sign in terms:
            for perm, parity in self._antisymmetric_permutations_4(indices):
                psi[perm] = sign * parity
                
        return psi
    
    def _antisymmetric_permutations(self, indices: Tuple[int, int, int]):
        """Generate all permutations of 3 indices with parities."""
        i, j, k = indices
        return [
            ((i, j, k), +1), ((j, k, i), +1), ((k, i, j), +1),
            ((j, i, k), -1), ((i, k, j), -1), ((k, j, i), -1),
        ]
    
    def _antisymmetric_permutations_4(self, indices: Tuple[int, int, int, int]):
        """Generate all permutations of 4 indices with parities (subset for efficiency)."""
        # Just do even/odd permutations
        i, j, k, l = indices
        return [
            ((i, j, k, l), +1), ((j, i, l, k), +1), ((k, l, i, j), +1), ((l, k, j, i), +1),
            ((j, i, k, l), -1), ((i, j, l, k), -1), ((l, k, i, j), -1), ((k, l, j, i), -1),
        ]
    
    def update_radius(self, new_radius: float):
        """Update the compactification radius."""
        self.radius = new_radius
        self.g = np.eye(self.dim) * (new_radius ** 2)
    
    def metric_at(self, y: np.ndarray) -> np.ndarray:
        """
        Compute metric tensor at point y in the fiber.
        
        For a true G2 manifold, the metric is determined by φ via:
        g_ij = (1/144) * φ_ikl * φ_jmn * ε^{klmnpqr} * φ_pqr
        
        We use a simplified version: flat metric with G2-invariant perturbations.
        """
        g = self.g.copy()
        
        # Add subtle position-dependent perturbations respecting G2 structure
        if np.linalg.norm(y) > 1e-10:
            norm_y = np.linalg.norm(y)
            y_hat = y / norm_y
            
            # Contract φ with y to get a 2-form, symmetrize for metric perturbation
            phi_y = np.einsum('ijk,k->ij', self.phi, y_hat)
            delta_g = 0.01 * self.radius**2 * np.tanh(norm_y) * (phi_y @ phi_y.T)
            g = g + delta_g
            
        return g
    
    def project_to_g2(self, matrix: np.ndarray) -> np.ndarray:
        """
        Project a 7x7 matrix onto the Lie algebra g2 ⊂ so(7).
        
        The Lie algebra g2 is 14-dimensional, consisting of antisymmetric
        matrices that preserve the 3-form φ under the Lie derivative.
        
        so(7) is 21-dimensional, so g2 is a proper subalgebra.
        """
        # First antisymmetrize (project to so(7))
        A = (matrix - matrix.T) / 2
        
        # Project onto g2 by removing components that don't preserve φ
        # A matrix X is in g2 iff: X^i_m φ^{mjk} + X^j_m φ^{imk} + X^k_m φ^{ijm} = 0
        
        for _ in range(5):  # Iterative projection
            # Compute violation of G2 constraint
            violation = (
                np.einsum('im,mjk->ijk', A, self.phi) +
                np.einsum('jm,imk->ijk', A, self.phi) +
                np.einsum('km,ijm->ijk', A, self.phi)
            )
            
            # Correction: contract violation back with φ
            correction = np.einsum('ijk,ljk->il', violation, self.phi) / 18
            correction = (correction - correction.T) / 2
            
            A = A - 0.3 * correction
            
        return A
    
    def g2_invariant_transform(self, v: np.ndarray, generator: np.ndarray) -> np.ndarray:
        """Apply a G2-invariant infinitesimal transformation to vector v."""
        A = self.project_to_g2(generator)
        return v + A @ v


# ============================================================================
# FIBER BUNDLE
# ============================================================================

class FiberBundle:
    """
    Fiber bundle with G2 geometry on the 7D fiber.
    
    Structure: E → B
    - Total space E has dimension 11
    - Base B has dimension 4 (spacetime)
    - Fiber F has dimension 7 (G2 manifold)
    
    The connection tells us how to parallel transport in the fiber
    as we move along the base.
    """
    
    def __init__(self, config: MTheoryConfig):
        self.config = config
        self.base_dim = config.base_dim
        self.fiber_dim = config.fiber_dim
        self.fiber_radius = config.fiber_radius
        
        if config.use_g2_holonomy:
            self.geometry = G2Geometry(dim=7, radius=config.fiber_radius)
        else:
            self.geometry = None
            
    def update_radius(self, new_radius: float):
        """Update fiber radius (for adaptive compactification)."""
        self.fiber_radius = new_radius
        if self.geometry:
            self.geometry.update_radius(new_radius)
    
    def split(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split 11D coordinates into 4D base + 7D fiber."""
        return z[..., :self.base_dim], z[..., self.base_dim:]
    
    def combine(self, base: np.ndarray, fiber: np.ndarray) -> np.ndarray:
        """Combine 4D base + 7D fiber into 11D coordinates."""
        return np.concatenate([base, fiber], axis=-1)
    
    def compactify(self, fiber: np.ndarray) -> np.ndarray:
        """
        Enforce compactification: fiber lives on a torus of radius r.
        
        More gentle than before - soft squashing rather than hard wrapping.
        """
        # Soft compactification using tanh-like squashing
        scale = self.fiber_radius * np.pi
        return scale * np.tanh(fiber / scale)
    
    def compactification_loss(self, fiber: np.ndarray) -> float:
        """
        Soft penalty for fiber coordinates being "too large".
        
        Uses a smooth potential that grows gently outside the radius.
        """
        # Effective "size" of fiber coordinates
        fiber_norm = np.sqrt(np.mean(fiber ** 2, axis=-1) + 1e-8)
        
        # Soft penalty: grows quadratically outside radius, flat inside
        target = self.fiber_radius
        excess = np.maximum(0, fiber_norm - target)
        
        return np.mean(excess ** 2) * self.config.compactification_strength


# ============================================================================
# NETWORK LAYERS
# ============================================================================

class MTheoryLayer:
    """
    Neural network layer respecting fiber bundle structure.
    
    Key insight: base and fiber are treated differently.
    - Base: unconstrained transformations (large-scale semantics)
    - Fiber: G2-constrained transformations (fine-grained, compact)
    - Cross-talk: controlled interaction between scales
    """
    
    def __init__(self, config: MTheoryConfig, hidden_dim: int = 64):
        self.config = config
        self.bundle = FiberBundle(config)
        self.hidden_dim = hidden_dim
        
        # Base pathway (unconstrained, large scale)
        self.W_base_1 = self._glorot_init(config.base_dim, hidden_dim)
        self.W_base_2 = self._glorot_init(hidden_dim, config.base_dim)
        self.b_base = np.zeros(config.base_dim)
        
        # Fiber pathway (G2-constrained, compact)
        self.W_fiber_g2 = self._glorot_init(config.fiber_dim, config.fiber_dim) * 0.5
        self.W_fiber_1 = self._glorot_init(config.fiber_dim, hidden_dim) * config.fiber_radius
        self.W_fiber_2 = self._glorot_init(hidden_dim, config.fiber_dim) * config.fiber_radius
        self.b_fiber = np.zeros(config.fiber_dim)
        
        # Cross-talk (controlled scale interaction)
        self.W_base2fiber = self._glorot_init(config.base_dim, config.fiber_dim) * config.fiber_radius * 0.5
        self.W_fiber2base = self._glorot_init(config.fiber_dim, config.base_dim) * config.fiber_radius * 0.5
        
        # Cache for backprop
        self._cache = {}
        
    def _glorot_init(self, fan_in: int, fan_out: int) -> np.ndarray:
        """Glorot/Xavier initialization."""
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_in, fan_out) * scale
    
    def _g2_constrain_fiber_weights(self) -> np.ndarray:
        """Get G2-projected fiber transformation."""
        if self.config.use_g2_holonomy and self.bundle.geometry:
            return self.bundle.geometry.project_to_g2(self.W_fiber_g2)
        return self.W_fiber_g2
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation (smoother than ReLU)."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Forward pass through the layer."""
        base, fiber = self.bundle.split(z)
        
        self._cache = {
            'z_in': z.copy(),
            'base_in': base.copy(),
            'fiber_in': fiber.copy()
        }
        
        # === BASE PATHWAY ===
        base_h = self._gelu(base @ self.W_base_1)
        base_out = base_h @ self.W_base_2 + self.b_base
        
        # === FIBER PATHWAY ===
        # First apply G2-constrained rotation
        W_g2 = self._g2_constrain_fiber_weights()
        fiber_rot = fiber + fiber @ W_g2  # Residual connection
        
        # Then nonlinear transform
        fiber_h = self._gelu(fiber_rot @ self.W_fiber_1)
        fiber_out = fiber_h @ self.W_fiber_2 + self.b_fiber
        
        # === CROSS-TALK ===
        # Base influences fiber (large scale → fine detail)
        fiber_out = fiber_out + np.tanh(base @ self.W_base2fiber)
        
        # Fiber influences base (fine detail → large scale, suppressed)
        base_out = base_out + fiber @ self.W_fiber2base
        
        # === COMPACTIFICATION ===
        fiber_out = self.bundle.compactify(fiber_out)
        
        self._cache.update({
            'base_h': base_h,
            'base_out': base_out,
            'fiber_rot': fiber_rot,
            'fiber_h': fiber_h,
            'fiber_out': fiber_out
        })
        
        # Compute geometric losses
        losses = {
            'compactification': self.bundle.compactification_loss(fiber_out)
        }
        
        z_out = self.bundle.combine(base_out, fiber_out)
        return z_out, losses
    
    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        """Simplified backward pass with weight updates."""
        grad_base, grad_fiber = self.bundle.split(grad_out)
        
        # Update output layer weights
        base_h = self._cache.get('base_h', np.zeros_like(grad_base))
        fiber_h = self._cache.get('fiber_h', np.zeros_like(grad_fiber))
        
        if base_h.ndim > 1:
            self.W_base_2 -= lr * (base_h.T @ grad_base) / base_h.shape[0]
            self.W_fiber_2 -= lr * (fiber_h.T @ grad_fiber) / fiber_h.shape[0]
        else:
            self.W_base_2 -= lr * np.outer(base_h, grad_base)
            self.W_fiber_2 -= lr * np.outer(fiber_h, grad_fiber)
        
        self.b_base -= lr * np.mean(grad_base, axis=0) if grad_base.ndim > 1 else lr * grad_base
        self.b_fiber -= lr * np.mean(grad_fiber, axis=0) if grad_fiber.ndim > 1 else lr * grad_fiber
        
        # Propagate gradients (simplified)
        grad_base_in = grad_base @ self.W_base_2.T @ self.W_base_1.T
        grad_fiber_in = grad_fiber @ self.W_fiber_2.T @ self.W_fiber_1.T
        
        return self.bundle.combine(grad_base_in, grad_fiber_in)


# ============================================================================
# FULL NETWORK
# ============================================================================

class MTheoryNetwork:
    """
    Complete M-theoretic neural network.
    
    Features:
    - Projects input to 11D M-theory space
    - Applies M-theory layers with G2-constrained fiber
    - Adaptive compactification radius
    - Tracks geometric diagnostics
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Optional[MTheoryConfig] = None,
        num_layers: int = 3,
        hidden_dim: int = 64
    ):
        self.config = config or MTheoryConfig()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Current fiber radius (can be adapted)
        self.current_radius = self.config.fiber_radius
        
        # Input projection: R^input_dim → R^11
        self.W_in = self._glorot_init(input_dim, self.config.total_dim)
        self.b_in = np.zeros(self.config.total_dim)
        
        # M-theory layers
        self.layers = [
            MTheoryLayer(self.config, hidden_dim)
            for _ in range(num_layers)
        ]
        
        # Output projection: R^11 → R^output_dim
        self.W_out = self._glorot_init(self.config.total_dim, output_dim)
        self.b_out = np.zeros(output_dim)
        
        # Training history
        self.history = {
            'loss': [], 'task_loss': [], 'comp_loss': [],
            'base_var': [], 'fiber_var': [], 'fiber_radius': [],
            'base_contribution': [], 'fiber_contribution': []
        }
        
        self._cache = {}
        
    def _glorot_init(self, fan_in: int, fan_out: int) -> np.ndarray:
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_in, fan_out) * scale
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Forward pass with full diagnostics."""
        self._cache = {'x': x.copy()}
        
        # Project to M-theory space
        z = x @ self.W_in + self.b_in
        self._cache['z_in'] = z.copy()
        
        bundle = FiberBundle(self.config)
        bundle.update_radius(self.current_radius)
        
        total_comp_loss = 0.0
        base_vars, fiber_vars = [], []
        
        # Apply M-theory layers
        for i, layer in enumerate(self.layers):
            layer.bundle.update_radius(self.current_radius)
            z, losses = layer.forward(z)
            total_comp_loss += losses['compactification']
            
            base, fiber = bundle.split(z)
            base_vars.append(np.var(base))
            fiber_vars.append(np.var(fiber))
        
        self._cache['z_final'] = z.copy()
        
        # Analyze base vs fiber contribution to output
        base_final, fiber_final = bundle.split(z)
        
        # Project to output
        y = z @ self.W_out + self.b_out
        
        # Estimate contributions
        base_contrib = np.var(base_final @ self.W_out[:self.config.base_dim, :])
        fiber_contrib = np.var(fiber_final @ self.W_out[self.config.base_dim:, :])
        total_contrib = base_contrib + fiber_contrib + 1e-8
        
        diagnostics = {
            'comp_loss': total_comp_loss,
            'base_var': base_vars,
            'fiber_var': fiber_vars,
            'base_contribution': base_contrib / total_contrib,
            'fiber_contribution': fiber_contrib / total_contrib,
            'fiber_radius': self.current_radius
        }
        
        return y, diagnostics
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray, diag: dict) -> Tuple[float, dict]:
        """Total loss with task and geometric components."""
        task_loss = np.mean((y_pred - y_true) ** 2)
        comp_loss = diag['comp_loss']
        
        total = task_loss + comp_loss
        
        return total, {
            'total': total,
            'task': task_loss,
            'comp': comp_loss
        }
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Backward pass with weight updates."""
        lr = self.config.learning_rate
        
        # Output gradient
        grad_y = 2 * (y_pred - y_true) / y_pred.size
        
        # Update output weights
        z_final = self._cache['z_final']
        if z_final.ndim > 1:
            self.W_out -= lr * (z_final.T @ grad_y) / z_final.shape[0]
        else:
            self.W_out -= lr * np.outer(z_final, grad_y)
        self.b_out -= lr * np.mean(grad_y, axis=0) if grad_y.ndim > 1 else lr * grad_y
        
        # Propagate through layers
        grad_z = grad_y @ self.W_out.T
        for layer in reversed(self.layers):
            grad_z = layer.backward(grad_z, lr)
        
        # Update input weights
        x = self._cache['x']
        if x.ndim > 1:
            self.W_in -= lr * (x.T @ grad_z) / x.shape[0]
        else:
            self.W_in -= lr * np.outer(x, grad_z)
        self.b_in -= lr * np.mean(grad_z, axis=0) if grad_z.ndim > 1 else lr * grad_z
    
    def adapt_radius(self, diagnostics: dict):
        """Adapt fiber radius based on usage."""
        if not self.config.adaptive_radius:
            return
        
        fiber_contrib = diagnostics['fiber_contribution']
        
        # If fiber contributing too little, expand it
        # If fiber contributing too much, compress it
        target_contrib = 0.2  # Fiber should contribute ~20%
        
        if fiber_contrib < target_contrib * 0.5:
            # Fiber underused, expand
            self.current_radius = min(
                self.current_radius * 1.01,
                self.config.max_fiber_radius
            )
        elif fiber_contrib > target_contrib * 2:
            # Fiber overused, compress
            self.current_radius = max(
                self.current_radius * 0.99,
                self.config.min_fiber_radius
            )
    
    def train(
        self,
        X: np.ndarray, Y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """Train the network."""
        n = X.shape[0]
        
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_shuf, Y_shuf = X[perm], Y[perm]
            
            epoch_metrics = []
            
            for i in range(0, n, batch_size):
                xb, yb = X_shuf[i:i+batch_size], Y_shuf[i:i+batch_size]
                
                # Forward
                y_pred, diag = self.forward(xb)
                
                # Loss
                _, losses = self.compute_loss(y_pred, yb, diag)
                
                # Backward
                self.backward(y_pred, yb)
                
                # Adapt radius
                self.adapt_radius(diag)
                
                epoch_metrics.append({**losses, **diag})
            
            # Record history
            avg = lambda k: np.mean([m[k] for m in epoch_metrics])
            self.history['loss'].append(avg('total'))
            self.history['task_loss'].append(avg('task'))
            self.history['comp_loss'].append(avg('comp'))
            self.history['base_var'].append(avg('base_var'))
            self.history['fiber_var'].append(avg('fiber_var'))
            self.history['fiber_radius'].append(self.current_radius)
            self.history['base_contribution'].append(avg('base_contribution'))
            self.history['fiber_contribution'].append(avg('fiber_contribution'))
            
            if verbose and (epoch + 1) % 10 == 0:
                bc = avg('base_contribution') * 100
                fc = avg('fiber_contribution') * 100
                print(f"Epoch {epoch+1:3d} | Loss: {avg('total'):.4f} | "
                      f"Task: {avg('task'):.4f} | "
                      f"Base: {bc:.1f}% | Fiber: {fc:.1f}% | "
                      f"Radius: {self.current_radius:.3f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_training(net: MTheoryNetwork, save_path: Optional[str] = None):
    """
    Create ASCII visualization of training dynamics.
    
    Shows:
    - Loss curves
    - Base vs fiber contribution over time
    - Fiber radius adaptation
    """
    h = net.history
    epochs = len(h['loss'])
    
    if epochs == 0:
        print("No training history to visualize!")
        return
    
    output = []
    output.append("=" * 70)
    output.append("M-THEORY NETWORK TRAINING VISUALIZATION")
    output.append("=" * 70)
    output.append("")
    
    # Loss curve (ASCII sparkline-ish)
    output.append("LOSS OVER TIME")
    output.append("-" * 70)
    
    losses = h['loss']
    min_l, max_l = min(losses), max(losses)
    range_l = max_l - min_l + 1e-8
    
    height = 10
    width = min(60, epochs)
    step = max(1, epochs // width)
    
    sampled = [losses[i] for i in range(0, epochs, step)][:width]
    
    for row in range(height, 0, -1):
        threshold = min_l + (row / height) * range_l
        line = ""
        for val in sampled:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        label = f"{threshold:.3f}" if row == height else ("" if row > 1 else f"{min_l:.3f}")
        output.append(f"{label:>8} |{line}|")
    
    output.append(f"{'':>8} +{'-' * width}+")
    output.append(f"{'':>8}  {'Epoch 1':<{width//2}}{'Epoch '+str(epochs):>{width//2}}")
    output.append("")
    
    # Base vs Fiber contribution
    output.append("BASE vs FIBER CONTRIBUTION")
    output.append("-" * 70)
    
    base_c = h['base_contribution']
    fiber_c = h['fiber_contribution']
    
    bar_width = 50
    for i, epoch_idx in enumerate(range(0, epochs, max(1, epochs // 10))):
        bc = base_c[epoch_idx]
        fc = fiber_c[epoch_idx]
        
        base_bars = int(bc * bar_width)
        fiber_bars = int(fc * bar_width)
        
        bar = "█" * base_bars + "░" * fiber_bars + " " * (bar_width - base_bars - fiber_bars)
        output.append(f"Epoch {epoch_idx+1:3d}: [{bar}] B:{bc*100:4.1f}% F:{fc*100:4.1f}%")
    
    output.append("")
    output.append("█ = Base contribution    ░ = Fiber contribution")
    output.append("")
    
    # Fiber radius evolution
    output.append("FIBER RADIUS ADAPTATION")
    output.append("-" * 70)
    
    radii = h['fiber_radius']
    min_r, max_r = min(radii), max(radii)
    
    if max_r - min_r > 0.001:
        output.append(f"Started at: {radii[0]:.4f}")
        output.append(f"Ended at:   {radii[-1]:.4f}")
        output.append(f"Range:      [{min_r:.4f}, {max_r:.4f}]")
        
        # Simple trend
        if radii[-1] > radii[0] * 1.05:
            output.append("Trend: ↑ Fiber expanded (needed more capacity)")
        elif radii[-1] < radii[0] * 0.95:
            output.append("Trend: ↓ Fiber compressed (too much capacity)")
        else:
            output.append("Trend: → Fiber stable")
    else:
        output.append(f"Constant at: {radii[0]:.4f}")
    
    output.append("")
    
    # Summary statistics
    output.append("FINAL STATISTICS")
    output.append("-" * 70)
    output.append(f"Final loss:              {h['loss'][-1]:.4f}")
    output.append(f"Final task loss:         {h['task_loss'][-1]:.4f}")
    output.append(f"Final base contribution: {h['base_contribution'][-1]*100:.1f}%")
    output.append(f"Final fiber contribution:{h['fiber_contribution'][-1]*100:.1f}%")
    output.append(f"Final fiber radius:      {h['fiber_radius'][-1]:.4f}")
    
    final_base_var = h['base_var'][-1]
    final_fiber_var = h['fiber_var'][-1]
    if isinstance(final_base_var, list):
        final_base_var = np.mean(final_base_var)
        final_fiber_var = np.mean(final_fiber_var)
    
    var_ratio = final_base_var / (final_fiber_var + 1e-8)
    output.append(f"Variance ratio (B/F):    {var_ratio:.1f}x")
    output.append("")
    
    # Interpretation
    output.append("INTERPRETATION")
    output.append("-" * 70)
    
    fc_final = h['fiber_contribution'][-1]
    if fc_final < 0.05:
        output.append("⚠ Fiber underutilized (<5%) - may need larger radius or different task")
    elif fc_final < 0.15:
        output.append("○ Fiber lightly used (5-15%) - capturing some fine structure")
    elif fc_final < 0.30:
        output.append("✓ Fiber well-utilized (15-30%) - good balance of scales")
    else:
        output.append("● Fiber heavily used (>30%) - may need more compactification")
    
    output.append("")
    output.append("=" * 70)
    
    result = "\n".join(output)
    print(result)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(result)
        print(f"\nVisualization saved to {save_path}")
    
    return result


# ============================================================================
# SYNTHETIC TASK
# ============================================================================

def create_multiscale_task(
    n_samples: int = 1000,
    input_dim: int = 16,
    output_dim: int = 8,
    coarse_weight: float = 0.7,
    fine_weight: float = 0.3
):
    """
    Create a task with explicit multi-scale structure.
    
    The output depends on:
    - Coarse features: linear combinations of inputs (should go to base)
    - Fine features: nonlinear/subtle patterns (should go to fiber)
    """
    np.random.seed(42)
    
    X = np.random.randn(n_samples, input_dim)
    
    # Coarse structure: simple linear relationships
    W_coarse = np.random.randn(input_dim, output_dim)
    Y_coarse = X @ W_coarse
    
    # Fine structure: nonlinear, high-frequency patterns
    W_fine = np.random.randn(input_dim, output_dim) * 0.3
    Y_fine = np.sin(3 * X @ W_fine) + 0.5 * np.cos(5 * X @ W_fine[::-1])
    
    # Combine
    Y = coarse_weight * Y_coarse + fine_weight * Y_fine
    
    # Normalize
    Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)
    
    return X, Y


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Full demonstration of M-theoretic network."""
    print("=" * 70)
    print("M-THEORETIC NEURAL NETWORK v3")
    print("4D Base (Spacetime) + 7D Fiber (G2 Manifold) = 11D M-Theory")
    print("=" * 70)
    print()
    
    # Configuration
    config = MTheoryConfig(
        base_dim=4,
        fiber_dim=7,
        fiber_radius=0.3,           # Start larger
        min_fiber_radius=0.05,
        max_fiber_radius=0.5,
        compactification_strength=0.5,  # Gentler
        adaptive_radius=True,
        use_g2_holonomy=True,
        ricci_flat_strength=0.01,
        learning_rate=0.01
    )
    
    print("Configuration:")
    print(f"  Base dimensions:  {config.base_dim} (spacetime)")
    print(f"  Fiber dimensions: {config.fiber_dim} (G2 manifold)")
    print(f"  Initial radius:   {config.fiber_radius}")
    print(f"  Adaptive radius:  {config.adaptive_radius}")
    print(f"  G2 holonomy:      {config.use_g2_holonomy}")
    print()
    
    # Create network
    net = MTheoryNetwork(
        input_dim=16,
        output_dim=8,
        config=config,
        num_layers=3,
        hidden_dim=64
    )
    
    # Create data
    print("Creating multi-scale synthetic task...")
    X, Y = create_multiscale_task(n_samples=1000, coarse_weight=0.7, fine_weight=0.3)
    
    X_train, X_test = X[:800], X[800:]
    Y_train, Y_test = Y[:800], Y[800:]
    
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Testing:  {X_test.shape[0]} samples")
    print(f"Task: 70% coarse structure + 30% fine structure")
    print()
    
    # Train
    print("Training...")
    print("-" * 70)
    net.train(X_train, Y_train, epochs=100, batch_size=32, verbose=True)
    print("-" * 70)
    print()
    
    # Test
    y_pred, diag = net.forward(X_test)
    test_mse = np.mean((y_pred - Y_test) ** 2)
    print(f"Test MSE: {test_mse:.4f}")
    print()
    
    # Visualize
    visualize_training(net)
    
    return net


if __name__ == "__main__":
    net = demo()
