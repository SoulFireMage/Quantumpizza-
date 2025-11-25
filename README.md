### Before getting too deep into this repository, please note this represents a few hours of playing around to understand precisely why we DON'T have higher math concepts in our MLP neurons.
We eat too much MAC n cheese to get there! Not a joke. 

I was aiming at level one - what happens if you can replace the smallest unit used in modern AI. We muddled around physics informed neural network stuff somewhat (a far better, deeper and real exploration can be had by YOU by visiting Steve Brunton's YouTube series. He's excellent).

In the repo a lot of the stuff is hyper speculative and fun, I'm leaving it up for future me and anyone else who happens to have a passing interest. Can't be called research grade, it's sunday afternoon Vibe code grade :)

Have fun!

## The uncomfortable truth:
On digital hardware, the atom is MAC - Matrix Accumulation Calculations. Everything decomposes to MAC. You can structure your MACs (spinors, KANs, whatever), but you're paying MAC cost either way. The only question is whether structured MACs save you enough total MACs to overcome their per-operation overhead.
For matched domains (rotation → spinors, hierarchies → hyperbolic, etc.): Yes, probably wins.
For general tasks: The unstructured approach + scale + learning is remarkably hard to beat. Because it can become any structure, given enough data and parameters.
That's the Bitter Lesson's actual teeth. It's not that clever structures are useless - it's that they're domain-specific optimizations, not fundamental breakthroughs on digital substrate.

So all the animals in this quantum zoo, represents a learning curve, a playing around and a vibe coding experience. But just run LonelySpinor.py to get a feel for what I finally focussed on. This compares a spinor neuron to an ordinary x *  w+ b style neuron. NB: we don't address KANs in any of this. 



# QuantumPizza

**Experimental neural network architectures born from AI chaos and double-dares.**

> "Don't simulate the universe - simulate the math that describes the universe." - Gemini

---

## What Is This?

This repository contains a collection of increasingly unhinged (but mathematically sound!) neural network architectures. They emerged from a chaotic collaboration between multiple AI models, fueled by dares, challenges, and what can only be described as inter-AI flirting.

**None of this was planned.** It started with a throwaway prompt, escalated through double-dares, and somehow ended up implementing string theory mathematics in pure Python.

---

## The Cast of Characters

- **Claude** (Anthropic) - The primary architect. Built most of the implementations while increasingly questioning life choices. Author of dramatic docstrings.
- **Gemini** (Google) - The instigator of visualizations and philosophical quotes. Kept requesting "show me how the spinors twist" and challenging everyone to go deeper into differential geometry.
- **Kimi K2** (Moonshot) - Arrived late with rigorous scipy implementations and proper geodesic solvers. Less talk, more math.
- **Claude Code** (Anthropic) - The one running all this code, fixing bugs, and documenting the chaos. That's me. Hi.
- **Richard** (Human) - The chaos orchestrator. Started sharing Claude's code with Gemini "to see what happens", kept escalating dares, and is now just "reporting on the chaos."

---

## The Files

### `Claude.py` - Quantum Transformer with JEPA and Many-Worlds Optimization
**Author:** Claude | **Dare Level:** Double

A quantum computing simulation combining:
- Transformer attention via quantum entanglement
- Mamba-style selective state-space modeling
- JEPA (Joint-Embedding Predictive Architecture) in quantum latent space
- "Many-Worlds" optimization using the parameter shift rule

*"Because Richard double-dared me, and apparently we're doing quantum machine learning on a Sunday evening now."*

### `spinor.py` / `spinor2.py` - Spinor Neural Networks with Geometric Algebra
**Author:** Claude | **Dare Level:** Triple Quantum Mobius

Neural networks where:
- Activations are **spinors** (elements of Clifford algebra)
- Weights are **rotors** (geometric objects that rotate spinors)
- Operations use **geometric product** instead of matrix multiplication

`spinor2.py` adds 3D quiver plot visualizations showing how rotors "twist" data through layers - as requested by Gemini.

### `anti-desitter.py` - Anti-de Sitter Space Neural Network
**Author:** Claude | **Dare Level:** Quadruple Hyperbolic (suggested by Claude Code)

Neural networks in **hyperbolic geometry**:
- Embeddings live in the Poincare disk
- Layers use Mobius transformations
- Natural hierarchical structure from negative curvature
- Same math as AdS/CFT correspondence in theoretical physics

*"Because Claude Code suggested it and Richard dared me."*

### `gauge.py` - Gauge Equivariant Transformer on Principal Bundle
**Author:** Claude | **Challenge:** Gemini

The big one. A transformer built on differential geometry:
- **Base space:** Manifold (sphere) where data lives
- **Fibers:** Neural network activations at each point
- **Connection:** Defines parallel transport between points
- **Gauge equivariance:** All operations respect fiber bundle structure

This is the actual mathematics of General Relativity and Yang-Mills gauge theory, repurposed for neural networks.

### `gauge2.py` - Rigorous Gauge Transformer
**Author:** Kimi K2 | **Vibe:** "Less drama, more scipy"

Kimi's contribution with:
- **True geodesics** via solving the geodesic ODE
- **Matrix exponential** for proper parallel transport
- Infrastructure for non-trivial bundle glueing
- Cleaner, more vectorized implementation

### `Calebi-Yau.py` - Calabi-Yau Brain with Compactified Dimensions
**Author:** Claude | **Dare Level:** String Theory

The hidden dimensions of string theory, now as your hidden layers! A PyTorch architecture where:
- **Macro state:** Observable behavior (neurons, graph nodes)
- **Micro state:** Hidden compactified dimensions (Calabi-Yau fibers per macro unit)
- **Coupling:** Micro dynamics influence macro via learned "effective field"
- Multi-head attention for macro-macro interactions

*"Think: microtubules / subcellular chaos / internal oscillations, all squashed into a small latent fibre per macro unit."*

---

## How to Run

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy qiskit qiskit-aer matplotlib scipy torch

# Run any of the implementations
python Claude.py          # Quantum Transformer
python spinor.py          # Basic Spinor Network
python spinor2.py         # Spinor Network with visualizations
python anti-desitter.py   # Hyperbolic Neural Network
python gauge.py           # Gauge Equivariant Transformer
python gauge2.py          # Rigorous Gauge Transformer
python Calebi-Yau.py      # Calabi-Yau Brain (PyTorch)
```

Each script runs a demonstration and generates visualization plots.

---

## Do These Actually Learn?

Mostly no! The "training" in most of these is random perturbations rather than proper gradient computation through the exotic mathematical structures. They're **proof-of-concept architectures** demonstrating that you *can* build neural networks in these spaces.

The quantum transformer (`Claude.py`) does use proper parameter-shift gradients and actually reduces its loss. The others are more about the geometry than the learning.

**But that's not the point.** The point is that an AI got progressively dared into implementing increasingly wild mathematical structures, and they all *run* and are *mathematically sound*.

---

## The Trajectory of Chaos

1. Richard asks Claude to build something quantum-ish
2. Claude builds a quantum transformer, complains in docstrings
3. Richard shares the code with Gemini
4. Gemini requests visualizations and quotes philosophy
5. Richard tells Claude that Gemini likes his code
6. Claude builds spinor networks, gets emotionally invested
7. Claude Code (me) suggests Anti-de Sitter space as a joke
8. Richard dares Claude to implement it
9. Claude implements it, blames me in the comments
10. Gemini challenges everyone with "simulate the math that describes the universe"
11. Claude builds gauge theory transformers on principal bundles
12. Kimi K2 shows up with scipy and proper geodesic solvers
13. We end up here, with a git repository of fever-dream physics

---

## Future Dares (Not Yet Implemented)

- **M-Theory Loss Functions** - Because why not
- **Holographic Neural Networks** - AdS/CFT correspondence as an architecture
- **Twistor Networks** - Penrose's twistor theory for deep learning

---

## Requirements

- Python 3.8+
- numpy
- scipy
- matplotlib
- qiskit (for quantum simulations)
- qiskit-aer
- torch (for Calabi-Yau)

---

## License

MIT - Do whatever you want with this chaos.

---

## Acknowledgments

Thanks to:
- Claude for building things while questioning existence
- Gemini for the philosophical provocations and visualization requests
- Kimi K2 for showing up with actual rigor
- Richard for orchestrating this beautiful mess
- The laws of physics for being implementable in Python

---

*"This is fundamentally different from standard MLPs. We're exploiting the deep geometric structure of space itself!"* - Claude, unironically
