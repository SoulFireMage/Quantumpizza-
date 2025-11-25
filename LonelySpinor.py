#!/usr/bin/env python3
"""
The Lonely SPINOR Neuron
========================

The geometric primitive, side-by-side with the scalar primitive.

SCALAR NEURON:     y = σ(w·x + b)
SPINOR NEURON:     y = normalize(R·x·R† + b)

Same SHAPE of computation, different PRIMITIVE operation.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PART 1: THE SCALAR NEURON (for comparison)
# =============================================================================

def scalar_neuron(inputs, weights, bias):
    """
    Classic neuron.
    
    Operations:
        1. MULTIPLY:     xᵢ × wᵢ        (scalar times scalar)
        2. ACCUMULATE:   Σ(xᵢ × wᵢ)     (sum to scalar)
        3. ADD BIAS:     z + b          (scalar + scalar)
        4. SQUASH:       tanh(z)        (scalar → scalar)
    
    Everything is SCALARS. Weights are SCALARS.
    """
    z = np.dot(inputs, weights) + bias
    return np.tanh(z)


# =============================================================================
# PART 2: CLIFFORD ALGEBRA PRIMITIVES (the new "multiply")
# =============================================================================

# Cl(3,0) multiplication table - precomputed
# M[i,j,k] = sign when basis[i] * basis[j] contributes to basis[k]
def build_geometric_product_table():
    """Build the 8×8×8 multiplication tensor for Cl(3,0)."""
    M = np.zeros((8, 8, 8))
    signature = [1, 1, 1]
    
    for i in range(8):
        for j in range(8):
            # Compute sign and result index
            sign = 1.0
            result = i ^ j  # XOR gives result blade
            
            for k in range(3):
                if j & (1 << k):
                    bits_to_pass = bin(i >> (k + 1)).count('1')
                    if bits_to_pass % 2 == 1:
                        sign *= -1
                    if i & (1 << k):
                        sign *= signature[k]
            
            M[i, j, result] = sign
    
    return M

_M = build_geometric_product_table()


def geometric_product(a, b):
    """
    THE NEW "MULTIPLY" - Geometric Product.
    
    Unlike scalar multiply (1 number × 1 number → 1 number),
    this is: 8 numbers × 8 numbers → 8 numbers
    
    But it's STRUCTURED. It encodes rotation, reflection,
    projection all in one operation.
    
    a ∗ b combines:
      - Inner product (dot): contracts, finds alignment
      - Outer product (wedge): extends, finds perpendicular
    
    For vectors: a∗b = a·b + a∧b
    """
    return np.einsum('ijk,i,j->k', _M, a, b)


def reverse(mv):
    """
    Reverse (dagger) - flip the order of basis vectors.
    
    For a rotor R, the reverse R† is its inverse (for unit rotors).
    """
    # Signs for Cl(3,0): grades 0,1,1,2,1,2,2,3
    # Pattern: (-1)^(k(k-1)/2) = [+,+,+,-,+,-,-,-]
    signs = np.array([1., 1., 1., -1., 1., -1., -1., -1.])
    return mv * signs


def exp_bivector(B):
    """
    THE NEW "WEIGHT TRANSFORM" - Exponential of bivector.
    
    Takes a bivector (oriented plane) and produces a ROTOR.
    
    exp(B) = cos(|B|) + sin(|B|)/|B| × B
    
    This is Euler's formula generalized!
    The bivector encodes: WHICH PLANE and HOW MUCH to rotate.
    """
    # Extract just the bivector part (indices 3, 5, 6)
    B_biv = B * np.array([0., 0., 0., 1., 0., 1., 1., 0.])
    
    # B² gives us the magnitude info
    B_sq = geometric_product(B_biv, B_biv)[0]  # Scalar part
    
    theta = np.sqrt(abs(B_sq) + 1e-10)
    
    if theta < 1e-6:
        # Small angle: exp(B) ≈ 1 + B
        result = np.zeros(8)
        result[0] = 1.0
        result += B_biv
    else:
        # Full formula: cos(θ) + sin(θ)/θ × B
        result = np.zeros(8)
        result[0] = np.cos(theta)
        result += (np.sin(theta) / theta) * B_biv
    
    return result


def sandwich(R, x):
    """
    THE NEW "TRANSFORM" - Sandwich Product.
    
    x' = R · x · R†
    
    This ROTATES x by the rotation encoded in R.
    
    Unlike scalar multiply (scales), this ROTATES.
    Rotation preserves lengths, angles, handedness.
    """
    temp = geometric_product(R, x)
    result = geometric_product(temp, reverse(R))
    return result


def normalize(mv):
    """
    THE NEW "ACTIVATION" - Normalization.
    
    Projects back onto the unit spinor manifold.
    
    Unlike tanh (arbitrary squash), this is GEOMETRIC.
    It preserves the DIRECTION in spinor space,
    only adjusting magnitude.
    """
    norm_sq = geometric_product(mv, reverse(mv))[0]
    if abs(norm_sq) < 1e-10:
        return mv
    return mv / np.sqrt(abs(norm_sq))


# =============================================================================
# PART 3: THE LONELY SPINOR NEURON
# =============================================================================

def spinor_neuron_explicit(input_spinor, weight_bivector, bias_spinor):
    """
    A single SPINOR neuron - every operation separated.
    
    Args:
        input_spinor:    8-dim spinor (even subalgebra element)
        weight_bivector: 8-dim bivector (will become rotor)
        bias_spinor:     8-dim spinor
    
    Operations:
        1. EXP:          R = exp(B)           (bivector → rotor)
        2. GEO-PROD 1:   temp = R · x         (rotor × spinor)
        3. GEO-PROD 2:   rotated = temp · R†  (complete sandwich)
        4. ADD BIAS:     sum = rotated + b    (spinor + spinor)
        5. NORMALIZE:    y = norm(sum)        (project to manifold)
    
    Returns:
        output_spinor: 8-dim spinor
    """
    
    # Step 1: Convert bivector weight to rotor
    # (This is like "activating" the weight)
    rotor = exp_bivector(weight_bivector)
    print(f"  1. Weight bivector → Rotor")
    print(f"     B = {weight_bivector}")
    print(f"     R = exp(B) = {rotor}")
    
    # Step 2: First half of sandwich
    temp = geometric_product(rotor, input_spinor)
    print(f"  2. First geometric product: R · x")
    print(f"     temp = {temp}")
    
    # Step 3: Second half of sandwich (with reverse)
    rotor_reverse = reverse(rotor)
    rotated = geometric_product(temp, rotor_reverse)
    print(f"  3. Second geometric product: temp · R†")
    print(f"     R† = {rotor_reverse}")
    print(f"     rotated = {rotated}")
    
    # Step 4: Add bias
    with_bias = rotated + bias_spinor
    print(f"  4. Add bias")
    print(f"     with_bias = {with_bias}")
    
    # Step 5: Normalize (the "activation")
    output = normalize(with_bias)
    print(f"  5. Normalize (activation)")
    print(f"     output = {output}")
    
    return output


def spinor_neuron(input_spinor, weight_bivector, bias_spinor):
    """Compact version - same computation."""
    R = exp_bivector(weight_bivector)
    rotated = sandwich(R, input_spinor)
    return normalize(rotated + bias_spinor)


# =============================================================================
# PART 4: WITH MULTIPLE INPUTS (like a real layer)
# =============================================================================

def spinor_neuron_multi_input(input_spinors, weight_bivectors, bias_spinor):
    """
    Spinor neuron with multiple inputs (more realistic).
    
    SCALAR:   y = σ(Σᵢ wᵢ·xᵢ + b)
    SPINOR:   y = normalize(Σᵢ Rᵢ·xᵢ·Rᵢ† + b)
    
    Each input gets its OWN rotor transformation.
    Then we accumulate. Then normalize.
    """
    n_inputs = len(input_spinors)
    
    # Accumulate rotated inputs
    accumulated = np.zeros(8)
    for i in range(n_inputs):
        R = exp_bivector(weight_bivectors[i])
        rotated = sandwich(R, input_spinors[i])
        accumulated += rotated
    
    # Add bias and normalize
    return normalize(accumulated + bias_spinor)


# =============================================================================
# PART 5: COMPARISON DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("THE LONELY NEURON: SCALAR vs SPINOR")
    print("=" * 70)
    
    # --- SCALAR NEURON ---
    print("\n" + "─" * 70)
    print("SCALAR NEURON: y = tanh(w·x + b)")
    print("─" * 70)
    
    scalar_inputs = np.array([0.5, -0.3, 0.8])
    scalar_weights = np.array([0.4, 0.7, -0.2])
    scalar_bias = 0.1
    
    print(f"\nInputs (3 scalars):  {scalar_inputs}")
    print(f"Weights (3 scalars): {scalar_weights}")
    print(f"Bias (1 scalar):     {scalar_bias}")
    
    # Step by step
    dot = np.dot(scalar_inputs, scalar_weights)
    pre_act = dot + scalar_bias
    output = np.tanh(pre_act)
    
    print(f"\nOperations:")
    print(f"  1. DOT PRODUCT:  {scalar_inputs} · {scalar_weights} = {dot:.4f}")
    print(f"  2. ADD BIAS:     {dot:.4f} + {scalar_bias} = {pre_act:.4f}")
    print(f"  3. TANH:         tanh({pre_act:.4f}) = {output:.4f}")
    print(f"\nOutput: {output:.4f}  (1 scalar)")
    
    print(f"\nPrimitive operation: MULTIPLY-ACCUMULATE (MAC)")
    print(f"  xᵢ × wᵢ  then  Σ  →  single scalar")
    
    # --- SPINOR NEURON ---
    print("\n" + "─" * 70)
    print("SPINOR NEURON: y = normalize(R·x·R† + b)")
    print("─" * 70)
    
    # A spinor has 8 components (but lives in 4D even subalgebra)
    # Indices: 0=scalar, 3=e12, 5=e13, 6=e23
    spinor_input = np.array([0.8, 0., 0., 0.3, 0., -0.2, 0.4, 0.])
    # Even subalgebra: components at 0, 3, 5, 6
    
    # Bivector weight (only components 3, 5, 6 matter)
    weight_bivector = np.array([0., 0., 0., 0.2, 0., 0.3, -0.1, 0.])
    
    # Bias (also in even subalgebra)
    spinor_bias = np.array([0.1, 0., 0., 0.05, 0., 0.0, 0.02, 0.])
    
    print(f"\nInput spinor (8 components, 4 active):")
    print(f"  [scalar, -, -, e12, -, e13, e23, -]")
    print(f"  {spinor_input}")
    
    print(f"\nWeight bivector (8 components, 3 active):")
    print(f"  [-, -, -, e12, -, e13, e23, -]")
    print(f"  {weight_bivector}")
    
    print(f"\nBias spinor:")
    print(f"  {spinor_bias}")
    
    print(f"\nOperations:")
    output = spinor_neuron_explicit(spinor_input, weight_bivector, spinor_bias)
    
    print(f"\nOutput: {output}  (8 components, 4 active)")
    
    print(f"\nPrimitive operation: GEOMETRIC PRODUCT + SANDWICH")
    print(f"  R·x·R† where R = exp(bivector)")
    
    # --- COMPARISON ---
    print("\n" + "=" * 70)
    print("COMPARISON: WHAT'S DIFFERENT?")
    print("=" * 70)
    print("""
    ┌────────────────┬─────────────────────┬─────────────────────────┐
    │                │ SCALAR NEURON       │ SPINOR NEURON           │
    ├────────────────┼─────────────────────┼─────────────────────────┤
    │ Input          │ n scalars           │ 1 spinor (4D effective) │
    │ Weight         │ n scalars           │ 1 bivector (3D)         │
    │ Bias           │ 1 scalar            │ 1 spinor (4D effective) │
    │ Output         │ 1 scalar            │ 1 spinor (4D effective) │
    ├────────────────┼─────────────────────┼─────────────────────────┤
    │ Core operation │ MULTIPLY xᵢ × wᵢ    │ SANDWICH R·x·R†         │
    │ What it does   │ SCALE               │ ROTATE                  │
    │ Preserves      │ Nothing             │ Norm, angles, structure │
    ├────────────────┼─────────────────────┼─────────────────────────┤
    │ Activation     │ tanh (arbitrary)    │ normalize (geometric)   │
    │ What it does   │ Squash to (-1,1)    │ Project to unit sphere  │
    ├────────────────┼─────────────────────┼─────────────────────────┤
    │ # of MACs      │ n + 1               │ ~50-100 (for Cl(3,0))   │
    │ Structure      │ None (learned)      │ Built-in (rotation)     │
    └────────────────┴─────────────────────┴─────────────────────────┘
    
    THE KEY QUESTION:
    
    Spinor neuron uses MORE MACs per "neuron."
    But does it capture MORE MEANING per neuron?
    
    If the task involves rotation/orientation:
      - Scalar neuron must LEARN rotational structure
      - Spinor neuron has it BUILT IN
    
    The bet: Richer primitive = fewer neurons needed = less learning
    """)


def plot_comparison_diagram():
    """Visual comparison of both neurons."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # SCALAR NEURON
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('SCALAR NEURON\ny = tanh(w·x + b)', fontsize=14, fontweight='bold')
    
    # Inputs
    for i, y in enumerate([4.5, 3, 1.5]):
        circle = plt.Circle((1.5, y), 0.35, fill=False, color='blue', linewidth=2)
        ax.add_patch(circle)
        ax.text(1.5, y, f'x{i+1}', ha='center', va='center', fontsize=12)
        ax.annotate('', xy=(3.5, 3), xytext=(1.85, y),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
        ax.text(2.3, (y+3)/2 + 0.15, f'×w{i+1}', fontsize=10, color='purple')
    
    # Sum
    circle = plt.Circle((4, 3), 0.4, fill=True, facecolor='lightyellow', 
                        edgecolor='orange', linewidth=2)
    ax.add_patch(circle)
    ax.text(4, 3, 'Σ', ha='center', va='center', fontsize=16)
    
    # Bias
    ax.annotate('', xy=(4, 2.6), xytext=(4, 1.5),
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.text(4, 1.2, '+b', ha='center', fontsize=12, color='green')
    
    # Activation
    ax.annotate('', xy=(5.8, 3), xytext=(4.4, 3),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    rect = plt.Rectangle((5.8, 2.5), 1.2, 1, fill=True, facecolor='lightcoral',
                         edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    ax.text(6.4, 3, 'tanh', ha='center', va='center', fontsize=12)
    
    # Output
    ax.annotate('', xy=(8, 3), xytext=(7, 3),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    circle = plt.Circle((8.3, 3), 0.35, fill=False, color='red', linewidth=2)
    ax.add_patch(circle)
    ax.text(8.3, 3, 'y', ha='center', va='center', fontsize=12)
    
    ax.text(5, 5.5, 'scalar × scalar → scalar', ha='center', fontsize=11, 
            style='italic', color='gray')
    
    # SPINOR NEURON
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('SPINOR NEURON\ny = normalize(R·x·R† + b)', fontsize=14, fontweight='bold')
    
    # Input spinor (single, but rich)
    circle = plt.Circle((1.5, 3), 0.5, fill=True, facecolor='lightblue',
                        edgecolor='blue', linewidth=2)
    ax.add_patch(circle)
    ax.text(1.5, 3, 'x', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(1.5, 2.2, 'spinor\n(8-dim)', ha='center', fontsize=9, color='gray')
    
    # Bivector weight → exp → rotor
    ax.annotate('', xy=(3.2, 4), xytext=(1.5, 4.8),
               arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
    rect = plt.Rectangle((3.2, 3.6), 1, 0.8, fill=True, facecolor='plum',
                         edgecolor='purple', linewidth=2)
    ax.add_patch(rect)
    ax.text(3.7, 4, 'exp', ha='center', va='center', fontsize=11)
    ax.text(2, 5, 'B\n(bivector)', ha='center', fontsize=9, color='purple')
    
    # Rotor
    ax.annotate('', xy=(4.5, 3.5), xytext=(4.2, 3.8),
               arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
    ax.text(4.8, 3.8, 'R', fontsize=12, color='purple', fontweight='bold')
    
    # Sandwich product
    ax.annotate('', xy=(4.2, 3), xytext=(2, 3),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ellipse = plt.matplotlib.patches.Ellipse((5, 3), 1.5, 1, fill=True, 
                                              facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(ellipse)
    ax.text(5, 3, 'R·x·R†', ha='center', va='center', fontsize=11)
    ax.text(5, 2.2, 'sandwich', ha='center', fontsize=9, color='gray')
    
    # Bias
    ax.annotate('', xy=(5, 2.5), xytext=(5, 1.5),
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.text(5, 1.2, '+b', ha='center', fontsize=12, color='green')
    ax.text(5.6, 1.2, '(spinor)', fontsize=9, color='gray')
    
    # Normalize
    ax.annotate('', xy=(7, 3), xytext=(5.75, 3),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    rect = plt.Rectangle((7, 2.5), 1.3, 1, fill=True, facecolor='lightcoral',
                         edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    ax.text(7.65, 3, 'norm', ha='center', va='center', fontsize=11)
    
    # Output
    ax.annotate('', xy=(9, 3), xytext=(8.3, 3),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    circle = plt.Circle((9.3, 3), 0.4, fill=True, facecolor='lightblue',
                        edgecolor='red', linewidth=2)
    ax.add_patch(circle)
    ax.text(9.3, 3, 'y', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(9.3, 2.3, 'spinor', ha='center', fontsize=9, color='gray')
    
    ax.text(5, 5.5, 'spinor × rotor × rotor† → spinor', ha='center', fontsize=11,
            style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('neuron_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: neuron_comparison.png")
    return fig


if __name__ == "__main__":
    demo()
    plot_comparison_diagram()
    plt.show()