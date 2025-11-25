#!/usr/bin/env python3
"""
Spinor Neural Network - JAX Accelerated (GPU)
==============================================

Fast implementation using:
- JAX autodiff (no numerical gradients!)
- JIT compilation
- GPU acceleration (NVIDIA)
- jaxga for Clifford algebra operations

Requirements:
    pip install jax[cuda12] jaxga matplotlib

For CPU-only (slower but works):
    pip install jax jaxga matplotlib

Author: Richard & Claude
Date: November 2025
Part of: QuantumPizza experimental architecture zoo
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
from typing import List, Tuple, NamedTuple
import time

# Check GPU availability
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# =============================================================================
# CLIFFORD ALGEBRA Cl(3,0) - JAX Implementation
# =============================================================================

class CliffordCl3:
    """
    Clifford Algebra Cl(3,0) optimized for JAX.
    
    Basis (8 elements):
        0: 1      (scalar)
        1: e1     (vector)
        2: e2     (vector)
        3: e12    (bivector)
        4: e3     (vector)
        5: e13    (bivector)
        6: e23    (bivector)
        7: e123   (pseudoscalar)
    
    Even subalgebra (spinors): indices 0, 3, 5, 6 (dim 4 = quaternions!)
    """
    
    def __init__(self):
        self.dim = 8
        self.n = 3
        
        # Precompute multiplication table (done once, on CPU)
        self.mult_sign, self.mult_idx = self._build_tables()
    
    def _build_tables(self):
        """Build geometric product tables."""
        import numpy as np
        sign = np.zeros((8, 8), dtype=np.float32)
        idx = np.zeros((8, 8), dtype=np.int32)
        
        signature = [1, 1, 1]  # Cl(3,0,0)
        
        for i in range(8):
            for j in range(8):
                s = 1.0
                result = i ^ j
                
                for k in range(3):
                    if j & (1 << k):
                        bits_to_pass = bin(i >> (k + 1)).count('1')
                        if bits_to_pass % 2 == 1:
                            s *= -1
                        if i & (1 << k):
                            s *= signature[k]
                
                sign[i, j] = s
                idx[i, j] = result
        
        return jnp.array(sign), jnp.array(idx)
    
    @partial(jit, static_argnums=(0,))
    def gp(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Geometric product using einsum for GPU efficiency.
        
        a, b: shape (8,) multivectors
        returns: shape (8,) result
        """
        # Outer product of coefficients
        outer = jnp.outer(a, b)  # (8, 8)
        
        # Apply signs
        signed = outer * self.mult_sign  # (8, 8)
        
        # Scatter-add to result indices
        result = jnp.zeros(8)
        for i in range(8):
            for j in range(8):
                result = result.at[self.mult_idx[i, j]].add(signed[i, j])
        
        return result
    
    @partial(jit, static_argnums=(0,))
    def gp_batch(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Batched geometric product: (batch, 8) x (batch, 8) -> (batch, 8)"""
        return vmap(self.gp)(a, b)
    
    @partial(jit, static_argnums=(0,))
    def reverse(self, mv: jnp.ndarray) -> jnp.ndarray:
        """Reverse (dagger): sign pattern for Cl(3,0)."""
        # Grades: 0,1,1,2,1,2,2,3 for indices 0-7
        # Sign = (-1)^(k(k-1)/2): +,+,+,-,+,-,-,-
        signs = jnp.array([1., 1., 1., -1., 1., -1., -1., -1.])
        return mv * signs
    
    @partial(jit, static_argnums=(0,))
    def grade_2(self, mv: jnp.ndarray) -> jnp.ndarray:
        """Project to grade 2 (bivectors): indices 3, 5, 6."""
        mask = jnp.array([0., 0., 0., 1., 0., 1., 1., 0.])
        return mv * mask
    
    @partial(jit, static_argnums=(0,))
    def even(self, mv: jnp.ndarray) -> jnp.ndarray:
        """Project to even subalgebra (spinors): indices 0, 3, 5, 6."""
        mask = jnp.array([1., 0., 0., 1., 0., 1., 1., 0.])
        return mv * mask
    
    @partial(jit, static_argnums=(0,))
    def norm_sq(self, mv: jnp.ndarray) -> jnp.ndarray:
        """Squared norm: <mv * reverse(mv)>_0"""
        return self.gp(mv, self.reverse(mv))[0]
    
    @partial(jit, static_argnums=(0,))
    def normalize(self, mv: jnp.ndarray) -> jnp.ndarray:
        """Normalize to unit magnitude."""
        n = jnp.sqrt(jnp.abs(self.norm_sq(mv)) + 1e-8)
        return mv / n
    
    @partial(jit, static_argnums=(0,))
    def exp_bivector(self, B: jnp.ndarray) -> jnp.ndarray:
        """
        exp(B) for bivector B -> rotor.
        
        For B² = -|B|²: exp(B) = cos(|B|) + sin(|B|)/|B| * B
        """
        B_biv = self.grade_2(B)
        B_sq = self.gp(B_biv, B_biv)[0]  # Scalar part of B²
        
        # B² should be negative for rotation
        theta = jnp.sqrt(jnp.abs(B_sq) + 1e-10)
        
        # Handle small theta (Taylor expansion)
        small = theta < 1e-6
        
        cos_t = jnp.where(small, 1.0 - theta**2/2, jnp.cos(theta))
        sinc_t = jnp.where(small, 1.0 - theta**2/6, jnp.sin(theta)/theta)
        
        # Sign flip if B² > 0 (shouldn't happen for pure bivector in Cl(3,0))
        sinc_t = jnp.where(B_sq > 0, jnp.sinh(theta)/theta, sinc_t)
        cos_t = jnp.where(B_sq > 0, jnp.cosh(theta), cos_t)
        
        result = jnp.zeros(8)
        result = result.at[0].set(cos_t)
        result = result + sinc_t * B_biv
        return result
    
    @partial(jit, static_argnums=(0,))
    def sandwich(self, R: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Sandwich product: R x R† (rotor application)."""
        return self.gp(self.gp(R, x), self.reverse(R))
    
    @partial(jit, static_argnums=(0,))
    def vec_to_spinor(self, v: jnp.ndarray) -> jnp.ndarray:
        """Embed 3D vector into spinor (scalar + bivector)."""
        return jnp.array([
            jnp.linalg.norm(v),  # scalar = magnitude
            0., 0.,
            v[0],  # e12
            0.,
            v[1],  # e13
            v[2],  # e23
            0.
        ])
    
    @partial(jit, static_argnums=(0,))
    def spinor_to_vec(self, s: jnp.ndarray) -> jnp.ndarray:
        """Extract 3D vector from spinor (bivector components)."""
        return jnp.array([s[3], s[5], s[6]])


# =============================================================================
# NETWORK PARAMETERS (as pytree-compatible NamedTuple)
# =============================================================================

class LayerParams(NamedTuple):
    """Parameters for one spinor layer."""
    bivectors: jnp.ndarray  # (out_dim, in_dim, 8)
    bias: jnp.ndarray       # (out_dim, 8)


def init_layer(key, in_dim: int, out_dim: int) -> LayerParams:
    """Initialize layer parameters."""
    k1, k2 = random.split(key)
    
    # Small bivectors -> rotors near identity
    bivectors = random.normal(k1, (out_dim, in_dim, 8)) * 0.1
    
    # Zero out non-bivector components (keep only indices 3, 5, 6)
    mask = jnp.array([0., 0., 0., 1., 0., 1., 1., 0.])
    bivectors = bivectors * mask
    
    # Small bias in even subalgebra
    bias = random.normal(k2, (out_dim, 8)) * 0.01
    even_mask = jnp.array([1., 0., 0., 1., 0., 1., 1., 0.])
    bias = bias * even_mask
    
    return LayerParams(bivectors, bias)


def init_network(key, hidden_dims: List[int]) -> List[LayerParams]:
    """Initialize full network."""
    dims = [1] + hidden_dims + [1]
    keys = random.split(key, len(dims) - 1)
    return [init_layer(keys[i], dims[i], dims[i+1]) 
            for i in range(len(dims) - 1)]


# =============================================================================
# FORWARD PASS (fully JIT-compiled)
# =============================================================================

def make_forward(ca: CliffordCl3):
    """Create forward function closed over Clifford algebra."""
    
    @jit
    def layer_forward(params: LayerParams, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Single layer forward pass.
        
        inputs: (batch, in_dim, 8)
        returns: (batch, out_dim, 8)
        """
        batch_size = inputs.shape[0]
        out_dim, in_dim = params.bivectors.shape[:2]
        
        outputs = jnp.zeros((batch_size, out_dim, 8))
        
        for i in range(out_dim):
            acc = jnp.zeros((batch_size, 8))
            for j in range(in_dim):
                # Compute rotor from bivector
                rotor = ca.exp_bivector(params.bivectors[i, j])
                # Apply rotor to each input in batch
                rotated = vmap(lambda x: ca.sandwich(rotor, x))(inputs[:, j])
                acc = acc + rotated
            
            # Add bias and normalize
            acc = acc + params.bias[i]
            acc = vmap(ca.normalize)(acc)
            outputs = outputs.at[:, i].set(acc)
        
        return outputs
    
    @jit
    def forward(params: List[LayerParams], vectors: jnp.ndarray) -> jnp.ndarray:
        """
        Full network forward pass.
        
        vectors: (batch, 3) input vectors
        returns: (batch, 3) output vectors
        """
        batch_size = vectors.shape[0]
        
        # Embed to spinors
        spinors = vmap(ca.vec_to_spinor)(vectors)
        spinors = spinors[:, None, :]  # (batch, 1, 8)
        
        # Pass through layers
        x = spinors
        for layer_params in params:
            x = layer_forward(layer_params, x)
        
        # Extract vectors
        return vmap(ca.spinor_to_vec)(x[:, 0])
    
    return forward


# =============================================================================
# LOSS AND TRAINING (with autodiff!)
# =============================================================================

def make_loss_fn(forward_fn):
    """Create loss function."""
    
    @jit
    def loss_fn(params, inputs, targets):
        preds = forward_fn(params, inputs)
        return jnp.mean((preds - targets) ** 2)
    
    return loss_fn


def make_train_step(forward_fn, lr: float = 0.01):
    """Create training step with autodiff gradients."""
    loss_fn = make_loss_fn(forward_fn)
    
    @jit
    def train_step(params, inputs, targets):
        loss, grads = value_and_grad(loss_fn)(params, inputs, targets)
        
        # SGD update
        new_params = []
        for p, g in zip(params, grads):
            new_bivectors = p.bivectors - lr * g.bivectors
            new_bias = p.bias - lr * g.bias
            
            # Re-project to bivector/even subspace
            biv_mask = jnp.array([0., 0., 0., 1., 0., 1., 1., 0.])
            even_mask = jnp.array([1., 0., 0., 1., 0., 1., 1., 0.])
            
            new_bivectors = new_bivectors * biv_mask
            new_bias = new_bias * even_mask
            
            new_params.append(LayerParams(new_bivectors, new_bias))
        
        return new_params, loss
    
    return train_step


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_rotation_data(key, n_samples: int, 
                           angle: float = jnp.pi/4) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate rotation prediction data (45° around z-axis)."""
    
    # Rotation matrix
    c, s = jnp.cos(angle), jnp.sin(angle)
    R = jnp.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    
    # Random unit vectors
    inputs = random.normal(key, (n_samples, 3))
    inputs = inputs / jnp.linalg.norm(inputs, axis=1, keepdims=True)
    
    # Targets
    targets = (R @ inputs.T).T
    
    return inputs, targets


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_results(inputs, targets, predictions, n_show: int = 15):
    """Visualize rotation results."""
    import numpy as np
    
    # Convert to numpy for matplotlib
    inputs = np.array(inputs[:n_show])
    targets = np.array(targets[:n_show])
    predictions = np.array(predictions[:n_show])
    
    fig = plt.figure(figsize=(14, 5))
    
    # Input vs Target
    ax1 = fig.add_subplot(131, projection='3d')
    for i in range(len(inputs)):
        ax1.quiver(0, 0, 0, inputs[i,0], inputs[i,1], inputs[i,2],
                   color='blue', alpha=0.6, arrow_length_ratio=0.1)
        ax1.quiver(0, 0, 0, targets[i,0], targets[i,1], targets[i,2],
                   color='green', alpha=0.6, arrow_length_ratio=0.1)
    ax1.set_title('Input (blue) → Target (green)')
    ax1.set_xlim(-1.2, 1.2); ax1.set_ylim(-1.2, 1.2); ax1.set_zlim(-1.2, 1.2)
    
    # Prediction vs Target
    ax2 = fig.add_subplot(132, projection='3d')
    for i in range(len(inputs)):
        ax2.quiver(0, 0, 0, predictions[i,0], predictions[i,1], predictions[i,2],
                   color='red', alpha=0.6, arrow_length_ratio=0.1)
        ax2.quiver(0, 0, 0, targets[i,0], targets[i,1], targets[i,2],
                   color='green', alpha=0.6, arrow_length_ratio=0.1)
    ax2.set_title('Prediction (red) vs Target (green)')
    ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2); ax2.set_zlim(-1.2, 1.2)
    
    # Error vectors
    ax3 = fig.add_subplot(133, projection='3d')
    errors = predictions - targets
    for i in range(len(inputs)):
        ax3.quiver(targets[i,0], targets[i,1], targets[i,2],
                   errors[i,0], errors[i,1], errors[i,2],
                   color='purple', alpha=0.6, arrow_length_ratio=0.3)
    ax3.set_title('Error vectors (from target)')
    ax3.set_xlim(-1.2, 1.2); ax3.set_ylim(-1.2, 1.2); ax3.set_zlim(-1.2, 1.2)
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--hidden', type=int, nargs='+', default=[8, 8])
    parser.add_argument('--train', type=int, default=200)
    parser.add_argument('--test', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 70)
    print("SPINOR NEURAL NETWORK - JAX GPU Accelerated")
    print("=" * 70)
    print(f"Device: {jax.devices()[0]}")
    print(f"Config: hidden={args.hidden}, epochs={args.epochs}, lr={args.lr}")
    print(f"Data: train={args.train}, test={args.test}")
    
    # Initialize
    key = random.PRNGKey(args.seed)
    k1, k2, k3 = random.split(key, 3)
    
    ca = CliffordCl3()
    params = init_network(k1, args.hidden)
    forward_fn = make_forward(ca)
    train_step = make_train_step(forward_fn, args.lr)
    loss_fn = make_loss_fn(forward_fn)
    
    # Data
    train_in, train_out = generate_rotation_data(k2, args.train)
    test_in, test_out = generate_rotation_data(k3, args.test)
    
    print(f"\nNetwork: {len(params)} layers")
    total_params = sum(p.bivectors.size + p.bias.size for p in params)
    print(f"Total parameters: {total_params}")
    
    # Warmup JIT
    print("\nWarming up JIT compilation...")
    t0 = time.time()
    _ = forward_fn(params, train_in[:10])
    _ = train_step(params, train_in[:10], train_out[:10])
    print(f"JIT warmup: {time.time() - t0:.2f}s")
    
    # Training
    print("\nTraining...")
    losses = []
    t0 = time.time()
    
    for epoch in range(args.epochs):
        params, loss = train_step(params, train_in, train_out)
        losses.append(float(loss))
        
        if (epoch + 1) % 20 == 0:
            test_loss = float(loss_fn(params, test_in, test_out))
            
            # Angular error
            preds = forward_fn(params, test_in)
            dots = jnp.clip(jnp.sum(preds * test_out, axis=1), -1, 1)
            angle_err = float(jnp.degrees(jnp.mean(jnp.arccos(dots))))
            
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:4d}: loss={loss:.6f}, test={test_loss:.6f}, "
                  f"err={angle_err:.2f}°, time={elapsed:.1f}s")
    
    total_time = time.time() - t0
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    test_preds = forward_fn(params, test_in)
    final_loss = float(loss_fn(params, test_in, test_out))
    
    dots = jnp.clip(jnp.sum(test_preds * test_out, axis=1), -1, 1)
    angles = jnp.arccos(dots)
    
    print(f"Final test loss: {final_loss:.6f}")
    print(f"Mean angular error: {float(jnp.degrees(jnp.mean(angles))):.2f}°")
    print(f"Max angular error: {float(jnp.degrees(jnp.max(angles))):.2f}°")
    print(f"Training time: {total_time:.2f}s ({args.epochs/total_time:.1f} epochs/sec)")
    
    # Visualizations
    print("\nSaving visualizations...")
    
    import numpy as np
    fig1, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (JAX GPU)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig1.savefig('spinor_jax_loss.png', dpi=150, bbox_inches='tight')
    print("  spinor_jax_loss.png")
    
    fig2 = visualize_results(test_in, test_out, test_preds)
    fig2.savefig('spinor_jax_rotation.png', dpi=150, bbox_inches='tight')
    print("  spinor_jax_rotation.png")
    
    plt.close('all')
    
    print("\n" + "=" * 70)
    print("ARCHITECTURE")
    print("=" * 70)
    print("""
  Clifford Algebra Cl(3,0):
    • 8-dimensional algebra
    • Spinors in even subalgebra (dim 4 = quaternions)
    • Rotors from exp(bivector)
    
  Network:
    • Weights are BIVECTORS → exp → ROTORS
    • Transform via sandwich product: R·x·R†
    • Autodiff through everything (no numerical gradients!)
    
  JAX acceleration:
    • JIT compiled forward/backward pass
    • GPU parallelism for batch operations
    • ~100-1000x faster than numpy numerical gradients
    
  Task: Learn 45° rotation around z-axis
  The network must discover that composition of rotors = rotation
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
