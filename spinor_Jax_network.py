#!/usr/bin/env python3
"""
Spinor Neural Network - JAX Accelerated (GPU) v2
=================================================

FIXED: Vectorized geometric product - no nested Python loops!
Uses einsum and segment_sum for GPU-friendly JIT compilation.

Requirements:
    pip install jax[cuda12] matplotlib

Author: Richard & Claude
Date: November 2025
Part of: QuantumPizza experimental architecture zoo
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
import jax.ops
import matplotlib.pyplot as plt
from functools import partial
from typing import List, Tuple, NamedTuple
import time
import numpy as np

# Check GPU
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")


# =============================================================================
# CLIFFORD ALGEBRA Cl(3,0) - VECTORIZED (no loops!)
# =============================================================================

def build_cl3_tables():
    """
    Build multiplication structure for Cl(3,0).
    Returns arrays that enable vectorized geometric product.
    
    Instead of iterating, we precompute a (8,8,8) tensor M where:
        (a * b)[k] = sum_{i,j where idx[i,j]==k} sign[i,j] * a[i] * b[j]
    
    This becomes: result = einsum('ijk,i,j->k', M, a, b)
    """
    sign = np.zeros((8, 8), dtype=np.float32)
    idx = np.zeros((8, 8), dtype=np.int32)
    signature = [1, 1, 1]
    
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
    
    # Build the (8,8,8) multiplication tensor
    # M[i,j,k] = sign[i,j] if idx[i,j] == k, else 0
    M = np.zeros((8, 8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            k = idx[i, j]
            M[i, j, k] = sign[i, j]
    
    return jnp.array(M), jnp.array(sign), jnp.array(idx)


# Precompute tables once at module load
_M, _SIGN, _IDX = build_cl3_tables()

# Grade of each basis element (popcount of index)
_GRADES = jnp.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=jnp.int32)

# Masks for grade projection
_GRADE2_MASK = jnp.array([0., 0., 0., 1., 0., 1., 1., 0.])  # Bivectors
_EVEN_MASK = jnp.array([1., 0., 0., 1., 0., 1., 1., 0.])    # Spinors (grade 0,2)

# Reverse signs: (-1)^(k(k-1)/2) for grade k
_REV_SIGNS = jnp.array([1., 1., 1., -1., 1., -1., -1., -1.])


@jit
def gp(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Geometric product - VECTORIZED.
    
    Uses einsum with precomputed multiplication tensor.
    No Python loops = happy JIT compiler.
    """
    return jnp.einsum('ijk,i,j->k', _M, a, b)


@jit
def gp_batch(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Batched geometric product: (batch,8) x (batch,8) -> (batch,8)"""
    return jnp.einsum('ijk,bi,bj->bk', _M, a, b)


@jit
def reverse(mv: jnp.ndarray) -> jnp.ndarray:
    """Reverse (dagger) operation."""
    return mv * _REV_SIGNS


@jit
def grade2(mv: jnp.ndarray) -> jnp.ndarray:
    """Project to bivectors (grade 2)."""
    return mv * _GRADE2_MASK


@jit
def even(mv: jnp.ndarray) -> jnp.ndarray:
    """Project to even subalgebra (spinors)."""
    return mv * _EVEN_MASK


@jit
def norm_sq(mv: jnp.ndarray) -> jnp.ndarray:
    """Squared norm."""
    return gp(mv, reverse(mv))[0]


@jit
def normalize(mv: jnp.ndarray) -> jnp.ndarray:
    """Normalize to unit magnitude."""
    n = jnp.sqrt(jnp.abs(norm_sq(mv)) + 1e-8)
    return mv / n


@jit
def exp_bivector(B: jnp.ndarray) -> jnp.ndarray:
    """
    exp(B) for bivector -> rotor.
    Vectorized, no branches in hot path.
    """
    B_biv = grade2(B)
    B_sq = gp(B_biv, B_biv)[0]
    
    theta = jnp.sqrt(jnp.abs(B_sq) + 1e-10)
    
    # Use jnp.where for branchless computation
    small = theta < 1e-6
    
    # Taylor expansion for small theta
    cos_t = jnp.where(small, 1.0 - theta**2/2, jnp.cos(theta))
    sinc_t = jnp.where(small, 1.0 - theta**2/6, jnp.sin(theta)/(theta + 1e-10))
    
    # Hyperbolic case if B² > 0 (shouldn't happen for pure bivector)
    hyp = B_sq > 1e-10
    cos_t = jnp.where(hyp, jnp.cosh(theta), cos_t)
    sinc_t = jnp.where(hyp, jnp.sinh(theta)/(theta + 1e-10), sinc_t)
    
    result = jnp.zeros(8).at[0].set(cos_t) + sinc_t * B_biv
    return result


@jit
def sandwich(R: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Sandwich product: R x R†"""
    return gp(gp(R, x), reverse(R))


@jit
def vec_to_spinor(v: jnp.ndarray) -> jnp.ndarray:
    """3D vector -> spinor embedding."""
    norm = jnp.linalg.norm(v)
    return jnp.array([norm, 0., 0., v[0], 0., v[1], v[2], 0.])


@jit
def spinor_to_vec(s: jnp.ndarray) -> jnp.ndarray:
    """Spinor -> 3D vector (bivector components)."""
    return jnp.array([s[3], s[5], s[6]])


# =============================================================================
# NETWORK PARAMETERS
# =============================================================================

class LayerParams(NamedTuple):
    bivectors: jnp.ndarray  # (out_dim, in_dim, 8)
    bias: jnp.ndarray       # (out_dim, 8)


def init_layer(key, in_dim: int, out_dim: int) -> LayerParams:
    k1, k2 = random.split(key)
    
    bivectors = random.normal(k1, (out_dim, in_dim, 8)) * 0.1 * _GRADE2_MASK
    bias = random.normal(k2, (out_dim, 8)) * 0.01 * _EVEN_MASK
    
    return LayerParams(bivectors, bias)


def init_network(key, hidden_dims: List[int]) -> List[LayerParams]:
    dims = [1] + hidden_dims + [1]
    keys = random.split(key, len(dims) - 1)
    return [init_layer(keys[i], dims[i], dims[i+1]) 
            for i in range(len(dims) - 1)]


# =============================================================================
# FORWARD PASS - VECTORIZED
# =============================================================================

@jit
def layer_forward(params: LayerParams, inputs: jnp.ndarray) -> jnp.ndarray:
    """
    Single layer: (batch, in_dim, 8) -> (batch, out_dim, 8)
    
    Vectorized over batch dimension.
    """
    batch_size = inputs.shape[0]
    out_dim, in_dim = params.bivectors.shape[:2]
    
    def process_output_unit(i):
        """Process one output unit across all batches."""
        def process_input(j):
            rotor = exp_bivector(params.bivectors[i, j])
            # Apply rotor to all batch elements
            return vmap(lambda x: sandwich(rotor, x))(inputs[:, j])
        
        # Sum over all input connections
        acc = jnp.zeros((batch_size, 8))
        for j in range(in_dim):
            acc = acc + process_input(j)
        
        return vmap(normalize)(acc + params.bias[i])
    
    # Stack outputs - using list comp (small fixed size, OK for JIT)
    outputs = jnp.stack([process_output_unit(i) for i in range(out_dim)], axis=1)
    return outputs


@jit 
def forward(params: List[LayerParams], vectors: jnp.ndarray) -> jnp.ndarray:
    """Full network: (batch, 3) -> (batch, 3)"""
    # Embed
    spinors = vmap(vec_to_spinor)(vectors)[:, None, :]
    
    # Layers
    x = spinors
    for p in params:
        x = layer_forward(p, x)
    
    # Extract
    return vmap(spinor_to_vec)(x[:, 0])


# =============================================================================
# LOSS AND TRAINING
# =============================================================================

@jit
def loss_fn(params, inputs, targets):
    preds = forward(params, inputs)
    return jnp.mean((preds - targets) ** 2)


@jit
def train_step(params, inputs, targets, lr):
    loss, grads = value_and_grad(loss_fn)(params, inputs, targets)
    
    new_params = []
    for p, g in zip(params, grads):
        new_biv = (p.bivectors - lr * g.bivectors) * _GRADE2_MASK
        new_bias = (p.bias - lr * g.bias) * _EVEN_MASK
        new_params.append(LayerParams(new_biv, new_bias))
    
    return new_params, loss


# =============================================================================
# DATA
# =============================================================================

def generate_rotation_data(key, n_samples: int, angle: float = jnp.pi/4):
    c, s = jnp.cos(angle), jnp.sin(angle)
    R = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    inputs = random.normal(key, (n_samples, 3))
    inputs = inputs / jnp.linalg.norm(inputs, axis=1, keepdims=True)
    targets = (R @ inputs.T).T
    
    return inputs, targets


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_results(inputs, targets, predictions, n_show: int = 15):
    inputs = np.array(inputs[:n_show])
    targets = np.array(targets[:n_show])
    predictions = np.array(predictions[:n_show])
    
    fig = plt.figure(figsize=(14, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    for i in range(len(inputs)):
        ax1.quiver(0, 0, 0, inputs[i,0], inputs[i,1], inputs[i,2],
                   color='blue', alpha=0.6, arrow_length_ratio=0.1)
        ax1.quiver(0, 0, 0, targets[i,0], targets[i,1], targets[i,2],
                   color='green', alpha=0.6, arrow_length_ratio=0.1)
    ax1.set_title('Input (blue) → Target (green)')
    ax1.set_xlim(-1.2, 1.2); ax1.set_ylim(-1.2, 1.2); ax1.set_zlim(-1.2, 1.2)
    
    ax2 = fig.add_subplot(132, projection='3d')
    for i in range(len(inputs)):
        ax2.quiver(0, 0, 0, predictions[i,0], predictions[i,1], predictions[i,2],
                   color='red', alpha=0.6, arrow_length_ratio=0.1)
        ax2.quiver(0, 0, 0, targets[i,0], targets[i,1], targets[i,2],
                   color='green', alpha=0.6, arrow_length_ratio=0.1)
    ax2.set_title('Prediction (red) vs Target (green)')
    ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2); ax2.set_zlim(-1.2, 1.2)
    
    ax3 = fig.add_subplot(133, projection='3d')
    errors = predictions - targets
    for i in range(len(inputs)):
        ax3.quiver(targets[i,0], targets[i,1], targets[i,2],
                   errors[i,0], errors[i,1], errors[i,2],
                   color='purple', alpha=0.6, arrow_length_ratio=0.3)
    ax3.set_title('Error vectors')
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
    print("SPINOR NEURAL NETWORK - JAX GPU (Vectorized v2)")
    print("=" * 70)
    print(f"Device: {jax.devices()[0]}")
    print(f"Config: hidden={args.hidden}, epochs={args.epochs}, lr={args.lr}")
    
    # Init
    key = random.PRNGKey(args.seed)
    k1, k2, k3 = random.split(key, 3)
    
    params = init_network(k1, args.hidden)
    train_in, train_out = generate_rotation_data(k2, args.train)
    test_in, test_out = generate_rotation_data(k3, args.test)
    
    total_params = sum(p.bivectors.size + p.bias.size for p in params)
    print(f"Network: {len(params)} layers, {total_params} parameters")
    
    # JIT warmup
    print("\nJIT warmup...")
    t0 = time.time()
    _ = forward(params, train_in[:10])
    _, _ = train_step(params, train_in[:10], train_out[:10], args.lr)
    jit_time = time.time() - t0
    print(f"JIT compilation: {jit_time:.2f}s")
    
    # Training
    print("\nTraining...")
    losses = []
    t0 = time.time()
    
    for epoch in range(args.epochs):
        params, loss = train_step(params, train_in, train_out, args.lr)
        losses.append(float(loss))
        
        if (epoch + 1) % 20 == 0:
            test_loss = float(loss_fn(params, test_in, test_out))
            preds = forward(params, test_in)
            dots = jnp.clip(jnp.sum(preds * test_out, axis=1), -1, 1)
            angle_err = float(jnp.degrees(jnp.mean(jnp.arccos(dots))))
            
            elapsed = time.time() - t0
            eps = (epoch + 1) / elapsed
            print(f"  Epoch {epoch+1:4d}: loss={loss:.6f}, test={test_loss:.6f}, "
                  f"err={angle_err:.2f}°  [{eps:.1f} ep/s]")
    
    total_time = time.time() - t0
    
    # Results
    print("\n" + "=" * 70)
    test_preds = forward(params, test_in)
    final_loss = float(loss_fn(params, test_in, test_out))
    dots = jnp.clip(jnp.sum(test_preds * test_out, axis=1), -1, 1)
    angles = jnp.arccos(dots)
    
    print(f"Final loss: {final_loss:.6f}")
    print(f"Angular error: mean={float(jnp.degrees(jnp.mean(angles))):.2f}°, "
          f"max={float(jnp.degrees(jnp.max(angles))):.2f}°")
    print(f"Time: {total_time:.2f}s ({args.epochs/total_time:.1f} epochs/sec)")
    print("=" * 70)
    
    # Save plots
    fig1, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, marker='o')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss (JAX GPU v2)')
    ax.set_yscale('log'); ax.grid(True, alpha=0.3)
    fig1.savefig('spinor_jax_v2_loss.png', dpi=150, bbox_inches='tight')
    print("Saved: spinor_jax_v2_loss.png")
    
    fig2 = visualize_results(test_in, test_out, test_preds)
    fig2.savefig('spinor_jax_v2_rotation.png', dpi=150, bbox_inches='tight')
    print("Saved: spinor_jax_v2_rotation.png")
    
    plt.close('all')


if __name__ == "__main__":
    main()