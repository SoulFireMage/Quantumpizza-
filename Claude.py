"""
Quantum Transformer with JEPA Architecture and Many-Worlds Optimization
========================================================================

A completely bonkers but theoretically sound implementation of:
1. Transformer attention via quantum entanglement
2. Mamba-style selective state-space modeling with quantum gates
3. JEPA predictive learning in quantum latent space
4. Many-Worlds optimization (measure in parallel universes without collapsing training state)

Author: Claude (under duress from Richard)
Status: Experimental / Slightly Mad
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator
from qiskit.quantum_info import state_fidelity, Statevector
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json


@dataclass
class QuantumState:
    """Represents a quantum state with its circuit and parameters."""
    circuit: QuantumCircuit
    params: Dict[str, float]
    statevector: Optional[np.ndarray] = None


class ManyWorldsOptimizer:
    """
    Optimizer that measures in parallel universes without collapsing the training state.
    
    Uses the parameter shift rule for gradient estimation:
    ∂⟨O⟩/∂θ = (⟨O⟩(θ+π/2) - ⟨O⟩(θ-π/2)) / 2
    """
    
    def __init__(self, learning_rate: float = 0.01, shots: int = 1024):
        self.learning_rate = learning_rate
        self.shots = shots
        self.backend = AerSimulator(method='statevector')
        self.reality_branches = []
        self.loss_history = []
        
    def measure_in_parallel_universe(self, circuit: QuantumCircuit) -> Dict[str, float]:
        """
        Execute measurement in a shadow universe (simulator run).
        The original quantum state remains uncollapsed in our training timeline.
        """
        # Create measurement circuit
        shadow_circuit = circuit.copy()
        shadow_circuit.measure_all()
        
        # Execute in shadow universe
        job = self.backend.run(shadow_circuit, shots=self.shots)
        counts = job.result().get_counts()
        
        # Normalize to probabilities
        total = sum(counts.values())
        probs = {k: v/total for k, v in counts.items()}
        
        return probs
    
    def compute_entropy_loss(self, measurement_probs: Dict[str, float]) -> float:
        """
        Compute entropy of measurement distribution.
        Lower entropy = more peaked distribution = better for classification.
        """
        entropy = -sum(p * np.log2(p + 1e-10) for p in measurement_probs.values())
        return entropy
    
    def compute_fidelity_loss(self, circuit1: QuantumCircuit, circuit2: QuantumCircuit) -> float:
        """
        Compute 1 - fidelity between two quantum states.
        Fidelity F = |⟨ψ₁|ψ₂⟩|²
        """
        # Get statevectors from both circuits
        sv1 = Statevector(circuit1)
        sv2 = Statevector(circuit2)
        
        # Compute fidelity
        fidelity = state_fidelity(sv1, sv2)
        loss = 1.0 - fidelity
        
        return loss
    
    def parameter_shift_gradient(
        self, 
        circuit_builder, 
        params: np.ndarray, 
        param_idx: int,
        loss_fn
    ) -> float:
        """
        Compute gradient using parameter shift rule.
        Each evaluation happens in a different parallel universe.
        """
        shift = np.pi / 2
        
        # Forward shift - parallel universe A
        params_plus = params.copy()
        params_plus[param_idx] += shift
        circuit_plus = circuit_builder(params_plus)
        loss_plus = loss_fn(circuit_plus)
        
        # Backward shift - parallel universe B
        params_minus = params.copy()
        params_minus[param_idx] -= shift
        circuit_minus = circuit_builder(params_minus)
        loss_minus = loss_fn(circuit_minus)
        
        # Gradient from the two parallel measurements
        gradient = (loss_plus - loss_minus) / 2.0
        
        # Store these realities for posterity
        self.reality_branches.append({
            'param_idx': param_idx,
            'shift': 'positive',
            'loss': loss_plus
        })
        self.reality_branches.append({
            'param_idx': param_idx,
            'shift': 'negative', 
            'loss': loss_minus
        })
        
        return gradient
    
    def step(self, circuit_builder, params: np.ndarray, loss_fn) -> np.ndarray:
        """
        Perform one optimization step across all parameters.
        """
        gradients = np.zeros_like(params)
        
        # Compute gradient for each parameter
        for i in range(len(params)):
            gradients[i] = self.parameter_shift_gradient(
                circuit_builder, params, i, loss_fn
            )
        
        # Update parameters
        new_params = params - self.learning_rate * gradients
        
        # Compute and store current loss
        current_circuit = circuit_builder(params)
        current_loss = loss_fn(current_circuit)
        self.loss_history.append(current_loss)
        
        return new_params


class QuantumAttentionHead:
    """
    Quantum attention mechanism using entanglement.
    
    In classical transformers: Attention(Q,K,V) = softmax(QK^T/√d)V
    In quantum version: Entanglement patterns encode attention, rotations are learnable
    """
    
    def __init__(self, n_qubits: int, name: str = "attn"):
        self.n_qubits = n_qubits
        self.name = name
        
        # Learnable parameters for Q, K, V transformations
        self.theta_q = ParameterVector(f'{name}_q', n_qubits)
        self.theta_k = ParameterVector(f'{name}_k', n_qubits)
        self.theta_v = ParameterVector(f'{name}_v', n_qubits)
        
    def build_circuit(self) -> QuantumCircuit:
        """
        Build quantum attention circuit.
        
        Architecture:
        1. Apply Q rotations (query transformation)
        2. Apply K rotations (key transformation)  
        3. Entangle Q and K qubits (this creates attention patterns)
        4. Apply V transformation via controlled rotations
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        circuit = QuantumCircuit(qr)
        
        # Query transformations
        for i in range(self.n_qubits):
            circuit.ry(self.theta_q[i], qr[i])
        
        # Key transformations
        for i in range(self.n_qubits):
            circuit.rz(self.theta_k[i], qr[i])
        
        # Entanglement layer (attention pattern)
        for i in range(self.n_qubits - 1):
            circuit.cx(qr[i], qr[i+1])
        # Wrap around
        circuit.cx(qr[self.n_qubits-1], qr[0])
        
        # Value transformations (controlled by attention)
        for i in range(self.n_qubits):
            target = (i + 1) % self.n_qubits
            circuit.cry(self.theta_v[i], qr[i], qr[target])
        
        return circuit
    
    def get_param_count(self) -> int:
        """Total number of learnable parameters."""
        return 3 * self.n_qubits


class QuantumMambaBlock:
    """
    Quantum version of Mamba's selective state-space model.
    
    Key idea: Use controlled quantum gates for selective information flow.
    Only certain qubits interact based on learned selection criteria.
    """
    
    def __init__(self, n_qubits: int, name: str = "mamba"):
        self.n_qubits = n_qubits
        self.name = name
        
        # Selection parameters (decide which qubits participate)
        self.sigma = ParameterVector(f'{name}_select', n_qubits)
        
        # State evolution parameters
        self.phi = ParameterVector(f'{name}_evolve', n_qubits)
        
    def build_circuit(self) -> QuantumCircuit:
        """
        Build selective state-space circuit.
        
        Architecture:
        1. Apply selection rotations (learned gating)
        2. Conditional evolution based on selection
        3. Selective entanglement (only between "selected" qubits)
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        circuit = QuantumCircuit(qr)
        
        # Selection layer
        for i in range(self.n_qubits):
            circuit.ry(self.sigma[i], qr[i])
        
        # Selective state evolution
        for i in range(self.n_qubits):
            target = (i + 1) % self.n_qubits
            # Controlled rotation - strength depends on selection
            circuit.cry(self.phi[i], qr[i], qr[target])
        
        # Selective entanglement pattern
        for i in range(0, self.n_qubits - 1, 2):
            circuit.cx(qr[i], qr[i+1])
        
        return circuit
    
    def get_param_count(self) -> int:
        return 2 * self.n_qubits


class QuantumJEPA:
    """
    Joint-Embedding Predictive Architecture in quantum space.
    
    Architecture:
    - Context encoder: Encodes input into quantum latent space
    - Target encoder: Encodes target (future/masked) into same latent space  
    - Predictor: Predicts target latent from context latent
    
    Loss: Minimize distance between predicted and actual target representations
    """
    
    def __init__(self, n_qubits: int, latent_qubits: int = None):
        self.n_qubits = n_qubits
        self.latent_qubits = latent_qubits or (n_qubits // 2)
        
        # Context encoder parameters
        self.theta_context = ParameterVector('ctx', n_qubits)
        
        # Target encoder parameters (can share some structure with context)
        self.theta_target = ParameterVector('tgt', n_qubits)
        
        # Predictor parameters
        self.theta_pred = ParameterVector('pred', self.latent_qubits)
        
    def build_context_encoder(self) -> QuantumCircuit:
        """
        Encode context into quantum latent representation.
        Uses local entanglement patterns (quantum convolutions).
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        circuit = QuantumCircuit(qr)
        
        # First layer: local rotations
        for i in range(self.n_qubits):
            circuit.ry(self.theta_context[i], qr[i])
        
        # Second layer: local entanglement (quantum convolution)
        for i in range(0, self.n_qubits - 1, 2):
            circuit.cx(qr[i], qr[i+1])
            circuit.rz(self.theta_context[i], qr[i])
        
        # Third layer: global entanglement for information mixing
        for i in range(self.n_qubits - 1):
            circuit.cx(qr[i], qr[i+1])
        
        return circuit
    
    def build_target_encoder(self) -> QuantumCircuit:
        """
        Encode target into quantum latent representation.
        Similar structure to context encoder but separate parameters.
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        circuit = QuantumCircuit(qr)
        
        # Similar architecture to context but with target parameters
        for i in range(self.n_qubits):
            circuit.ry(self.theta_target[i], qr[i])
        
        for i in range(0, self.n_qubits - 1, 2):
            circuit.cx(qr[i], qr[i+1])
            circuit.rz(self.theta_target[i], qr[i])
        
        for i in range(self.n_qubits - 1):
            circuit.cx(qr[i], qr[i+1])
        
        return circuit
    
    def build_predictor(self, context_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Predict target representation from context representation.
        """
        # Start with context encoding
        predictor = context_circuit.copy()
        
        # Add prediction-specific transformations
        qr = predictor.qregs[0]
        
        # Focus on latent qubits for prediction
        for i in range(self.latent_qubits):
            predictor.ry(self.theta_pred[i], qr[i])
        
        # Entangle predictions
        for i in range(self.latent_qubits - 1):
            predictor.cx(qr[i], qr[i+1])
        
        return predictor
    
    def get_param_count(self) -> int:
        return 2 * self.n_qubits + self.latent_qubits


class QuantumTransformerJEPA:
    """
    Complete Quantum Transformer with JEPA learning.
    
    Combines:
    - Quantum attention heads
    - Mamba selective state-space blocks
    - JEPA predictive learning objective
    - Many-worlds optimization
    """
    
    def __init__(
        self, 
        n_qubits: int = 6,
        n_attention_heads: int = 2,
        n_mamba_blocks: int = 1,
        latent_qubits: int = 3
    ):
        self.n_qubits = n_qubits
        self.n_attention_heads = n_attention_heads
        self.n_mamba_blocks = n_mamba_blocks
        self.latent_qubits = latent_qubits
        
        # Build architecture components
        self.attention_heads = [
            QuantumAttentionHead(n_qubits, f"attn_{i}") 
            for i in range(n_attention_heads)
        ]
        
        self.mamba_blocks = [
            QuantumMambaBlock(n_qubits, f"mamba_{i}") 
            for i in range(n_mamba_blocks)
        ]
        
        self.jepa = QuantumJEPA(n_qubits, latent_qubits)
        
        # Initialize parameters
        self.params = self._initialize_params()
        
        # Optimizer
        self.optimizer = ManyWorldsOptimizer(learning_rate=0.05, shots=1024)
        
    def _initialize_params(self) -> np.ndarray:
        """
        Initialize all parameters randomly.
        """
        total_params = (
            sum(head.get_param_count() for head in self.attention_heads) +
            sum(block.get_param_count() for block in self.mamba_blocks) +
            self.jepa.get_param_count()
        )
        
        # Random initialization with small values
        params = np.random.randn(total_params) * 0.1
        
        print(f"Total parameters: {total_params}")
        print(f"  - Attention heads: {sum(head.get_param_count() for head in self.attention_heads)}")
        print(f"  - Mamba blocks: {sum(block.get_param_count() for block in self.mamba_blocks)}")
        print(f"  - JEPA: {self.jepa.get_param_count()}")
        
        return params
    
    def build_full_circuit(self, params: np.ndarray) -> Tuple[QuantumCircuit, QuantumCircuit]:
        """
        Build complete quantum circuit with current parameters.
        Returns (context_circuit, target_circuit) for JEPA training.
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        context_circuit = QuantumCircuit(qr)
        
        param_idx = 0
        
        # Add attention heads
        for head in self.attention_heads:
            head_circuit = head.build_circuit()
            n_params = head.get_param_count()
            
            # Bind parameters
            param_dict = {
                param: params[param_idx + i] 
                for i, param in enumerate(head_circuit.parameters)
            }
            head_circuit = head_circuit.assign_parameters(param_dict)
            
            context_circuit.compose(head_circuit, inplace=True)
            param_idx += n_params
        
        # Add Mamba blocks
        for block in self.mamba_blocks:
            block_circuit = block.build_circuit()
            n_params = block.get_param_count()
            
            param_dict = {
                param: params[param_idx + i]
                for i, param in enumerate(block_circuit.parameters)
            }
            block_circuit = block_circuit.assign_parameters(param_dict)
            
            context_circuit.compose(block_circuit, inplace=True)
            param_idx += n_params
        
        # Build JEPA components
        jepa_context = self.jepa.build_context_encoder()
        jepa_target = self.jepa.build_target_encoder()
        
        # Bind JEPA parameters
        n_context_params = len(self.jepa.theta_context)
        n_target_params = len(self.jepa.theta_target)
        n_pred_params = len(self.jepa.theta_pred)
        
        context_dict = {
            self.jepa.theta_context[i]: params[param_idx + i]
            for i in range(n_context_params)
        }
        param_idx += n_context_params
        
        target_dict = {
            self.jepa.theta_target[i]: params[param_idx + i]
            for i in range(n_target_params)
        }
        param_idx += n_target_params
        
        pred_dict = {
            self.jepa.theta_pred[i]: params[param_idx + i]
            for i in range(n_pred_params)
        }
        
        jepa_context = jepa_context.assign_parameters(context_dict)
        jepa_target = jepa_target.assign_parameters(target_dict)
        
        # Combine with transformer output
        context_circuit.compose(jepa_context, inplace=True)
        
        # Build predictor from context
        predictor = self.jepa.build_predictor(context_circuit)
        predictor = predictor.assign_parameters(pred_dict)
        
        # Target circuit
        target_circuit = QuantumCircuit(qr)
        target_circuit.compose(jepa_target, inplace=True)
        
        return predictor, target_circuit
    
    def train_step(self) -> float:
        """
        Perform one training step using many-worlds optimization.
        """
        def loss_fn(params):
            predictor, target = self.build_full_circuit(params)
            loss = self.optimizer.compute_fidelity_loss(predictor, target)
            return loss
        
        # Update parameters
        self.params = self.optimizer.step(
            lambda p: self.build_full_circuit(p),
            self.params,
            lambda circuits: self.optimizer.compute_fidelity_loss(circuits[0], circuits[1])
        )
        
        return self.optimizer.loss_history[-1]
    
    def train(self, n_epochs: int = 20):
        """
        Train the quantum transformer.
        """
        print(f"\n{'='*60}")
        print(f"Training Quantum Transformer-JEPA")
        print(f"{'='*60}")
        print(f"Qubits: {self.n_qubits}")
        print(f"Attention Heads: {self.n_attention_heads}")
        print(f"Mamba Blocks: {self.n_mamba_blocks}")
        print(f"Latent Qubits: {self.latent_qubits}")
        print(f"Total Parameters: {len(self.params)}")
        print(f"{'='*60}\n")
        
        for epoch in range(n_epochs):
            loss = self.train_step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d} | Loss: {loss:.6f} | "
                      f"Branches explored: {len(self.optimizer.reality_branches)}")
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Final Loss: {self.optimizer.loss_history[-1]:.6f}")
        print(f"Total parallel universes explored: {len(self.optimizer.reality_branches)}")
        print(f"{'='*60}\n")
    
    def visualize_training(self):
        """
        Plot training loss curve.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.optimizer.loss_history, linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (1 - Fidelity)', fontsize=12)
        plt.title('Quantum Transformer-JEPA Training\n(Optimized Across Parallel Universes)', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('quantum_training_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Training visualization saved!")
    
    def get_circuit_visualization(self) -> str:
        """
        Get a text representation of the circuit architecture.
        """
        predictor, target = self.build_full_circuit(self.params)
        
        info = []
        info.append(f"\nQuantum Circuit Architecture:")
        info.append(f"{'='*60}")
        info.append(f"Total qubits: {self.n_qubits}")
        info.append(f"Circuit depth: {predictor.depth()}")
        info.append(f"Number of gates: {len(predictor.data)}")
        info.append(f"{'='*60}")
        
        return '\n'.join(info)


def demonstrate_quantum_transformer():
    """
    Demonstrate the complete quantum transformer system.
    """
    print("\n" + "="*70)
    print("QUANTUM TRANSFORMER WITH JEPA AND MANY-WORLDS OPTIMIZATION")
    print("="*70)
    print("\nBecause Richard double-dared me, and apparently we're doing")
    print("quantum machine learning on a Sunday evening now.")
    print("\n" + "="*70 + "\n")
    
    # Create model
    model = QuantumTransformerJEPA(
        n_qubits=6,
        n_attention_heads=2,
        n_mamba_blocks=1,
        latent_qubits=3
    )
    
    # Print architecture
    print(model.get_circuit_visualization())
    
    # Train
    model.train(n_epochs=20)
    
    # Visualize
    model.visualize_training()
    
    # Print some interesting statistics
    print("\nReality Branch Statistics:")
    print(f"  Total branches explored: {len(model.optimizer.reality_branches)}")
    print(f"  Average loss across branches: {np.mean([b['loss'] for b in model.optimizer.reality_branches]):.6f}")
    print(f"  Best loss found: {min([b['loss'] for b in model.optimizer.reality_branches]):.6f}")
    
    return model


if __name__ == "__main__":
    # Run the demonstration
    model = demonstrate_quantum_transformer()
    
    print("\n" + "="*70)
    print("COMPLETE! And it actually works!")
    print("(In theory. In a simulator. Across parallel universes.)")
    print("="*70 + "\n")