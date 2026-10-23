#!/usr/bin/env python3
"""
Rotation Experiment: does a richer primitive need fewer parameters?
==================================================================

Tests the bet in LonelySpinor.py: "richer primitive = fewer neurons".

A hidden rotation R* is applied to inputs. Each model has to learn it from
examples, and is then tested on inputs from a wedge it never saw.

TASK A (vectors):  v  →  R*·v·R*†     (the sandwich is the right action)
TASK B (spinors):  x  →  R*·x         (the one-sided product is the right action)

MODELS:
    rotor neuron     y = R·x·R†,  R = rotor(B)     3 parameters
    spinor neuron    y = R·x,     R = rotor(B)     3 parameters
    linear map       y = W·x, any d×d matrix        d² parameters (no geometry)
    tanh MLP         one hidden layer, width h      (2d+1)h + d parameters

A rotation is linear, so the linear map is the honest control: it can
represent the answer exactly, with no geometry built in. Whatever the
neurons gain over it comes from the constraint (only rotations allowed),
not from extra power.

Each model is trained from 3 random starts with full-batch Adam, and the
median test error is reported. Bias and normalize from LonelySpinor are
left out: the targets are unit-length already, and leaving them out keeps
the neurons at exactly 3 parameters.

What to look for:
    - Does a 3-parameter neuron match or beat MLPs with 10-100x more?
    - Does it still work on the held-out wedge, where the MLP has to guess?
    - Does the wrong action fail? (sandwich on Task B, one-sided on Task A)

RESULTS (256 clean training samples, median of 3 seeds):
    - The matching neuron is exact (MSE ~1e-25, rounding level) with 3
      parameters, on the held-out wedge too. The best MLP reaches ~1e-5
      with 60-600 parameters and is 2-6x worse on the wedge.
    - The wrong action fails at any training length: MSE ~0.4-0.7. The
      sandwich can't fit R*·x and the one-sided product can't fit R*·v·R*†.
    - The linear map is also exact, with 9 (Task A) or 16 (Task B)
      parameters. So against the honest control the gain is 3x-5x fewer
      parameters, not 100x. The MLP is a weak baseline for a linear task.

FEW, NOISY SAMPLES (--sweep; noise 0.05 on targets, clean wedge test,
median of 8 trials; about 5 minutes):
    Prediction, made before running: least-squares excess error grows like
    noise² × parameters / N, so the linear map should be ~3x worse on
    vectors (9 vs 3) and ~5x worse on spinors (16 vs 3), and far worse when
    N is below the dimension, where it is underdetermined.
    - N = 2-3: the neuron wins by 60x (vectors) and 200-300x (spinors).
    - N = 5-100: vectors 2.5-7x, close to the prediction. Spinors 7-20x,
      above it. A guess, not tested: the wedge test makes the linear map
      extrapolate, which costs it more in 4D.
    So the constraint pays exactly where it should: with little data. With
    plenty of clean data the advantage shrinks to the parameter ratio.
"""

import numpy as np

from LonelySpinor import _M, rotor

VEC = [1, 2, 4]          # e1, e2, e3
EVEN = [0, 3, 5, 6]      # 1, e12, e13, e23
REV = np.array([1., 1., 1., -1., 1., -1., -1., -1.])
SEEDS = [0, 1, 2]
WIDTHS = [1, 2, 4, 8, 16, 32, 64]
STEPS = 3000
LR = 0.01


# =============================================================================
# DATA
# =============================================================================

def hidden_rotor():
    """A fixed rotation of 100° in a tilted plane."""
    plane = np.array([0.3, 0.5, -0.8])
    plane /= np.linalg.norm(plane)
    B = np.zeros(8)
    B[[3, 5, 6]] = np.radians(100) * plane
    return rotor(B)


def left(R, X):
    """R·X for a batch X of shape (N, 8)."""
    return np.einsum('ijk,i,nj->nk', _M, R, X)


def right(X, R):
    """X·R for a batch X of shape (N, 8)."""
    return np.einsum('ijk,ni,j->nk', _M, X, R)


def make_task(kind, n, wedge, rng):
    """
    Unit inputs whose angle in the first two active components lies in the
    given wedge (degrees). Training uses [0, 270), testing uses [270, 360).
    """
    idx = VEC if kind == 'vector' else EVEN
    X = np.zeros((0, 8))
    while len(X) < n:
        c = rng.normal(size=(4 * n, len(idx)))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        ang = np.degrees(np.arctan2(c[:, 1], c[:, 0])) % 360
        c = c[(ang >= wedge[0]) & (ang < wedge[1])]
        batch = np.zeros((len(c), 8))
        batch[:, idx] = c
        X = np.vstack([X, batch])
    X = X[:n]

    R = hidden_rotor()
    if kind == 'vector':
        Y = right(left(R, X), R * REV)
    else:
        Y = left(R, X)
    return X, Y


# =============================================================================
# MODELS
# =============================================================================

def neuron_forward(B3, X, action):
    B = np.zeros(8)
    B[[3, 5, 6]] = B3
    R = rotor(B)
    if action == 'sandwich':
        return right(left(R, X), R * REV)
    return left(R, X)


def mse(P, Y):
    return np.mean(np.sum((P - Y) ** 2, axis=1))


def adam(params, grad_fn, steps=STEPS, lr=LR):
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for t in range(1, steps + 1):
        grads = grad_fn(params)
        for i, g in enumerate(grads):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            mh = m[i] / (1 - 0.9 ** t)
            vh = v[i] / (1 - 0.999 ** t)
            params[i] = params[i] - lr * mh / (np.sqrt(vh) + 1e-8)
    return params


def train_neuron(X, Y, action, seed):
    """3 parameters; gradient by central differences (cheap at this size)."""
    rng = np.random.default_rng(seed)
    B3 = rng.normal(scale=0.5, size=3)

    def grad_fn(params):
        b = params[0]
        g = np.zeros(3)
        for i in range(3):
            d = np.zeros(3)
            d[i] = 1e-5
            g[i] = (mse(neuron_forward(b + d, X, action), Y)
                    - mse(neuron_forward(b - d, X, action), Y)) / 2e-5
        return [g]

    (B3,) = adam([B3], grad_fn)
    return lambda Xt: neuron_forward(B3, Xt, action)


def train_mlp(X, Y, idx, width, seed):
    """One hidden tanh layer, linear output, analytic gradients."""
    rng = np.random.default_rng(seed)
    Xa, Ya = X[:, idx], Y[:, idx]
    d = len(idx)
    params = [rng.normal(scale=1 / np.sqrt(d), size=(d, width)),
              np.zeros(width),
              rng.normal(scale=1 / np.sqrt(width), size=(width, d)),
              np.zeros(d)]

    def grad_fn(p):
        W1, b1, W2, b2 = p
        H = np.tanh(Xa @ W1 + b1)
        P = H @ W2 + b2
        dP = 2 * (P - Ya) / len(Xa)
        dH = (dP @ W2.T) * (1 - H ** 2)
        return [Xa.T @ dH, dH.sum(0), H.T @ dP, dP.sum(0)]

    W1, b1, W2, b2 = adam(params, grad_fn)

    def predict(Xt):
        out = np.zeros_like(Xt)
        out[:, idx] = np.tanh(Xt[:, idx] @ W1 + b1) @ W2 + b2
        return out
    return predict


def train_linear(X, Y, idx):
    """The no-geometry control: any d×d matrix, fitted by least squares."""
    W, *_ = np.linalg.lstsq(X[:, idx], Y[:, idx], rcond=None)

    def predict(Xt):
        out = np.zeros_like(Xt)
        out[:, idx] = Xt[:, idx] @ W
        return out
    return predict


# =============================================================================
# RUN
# =============================================================================

def run_task(kind):
    rng = np.random.default_rng(42)
    idx = VEC if kind == 'vector' else EVEN
    d = len(idx)
    Xtr, Ytr = make_task(kind, 256, (0, 270), rng)
    Xin, Yin = make_task(kind, 512, (0, 270), rng)
    Xout, Yout = make_task(kind, 512, (270, 360), rng)

    rows = []

    def evaluate(name, n_params, trainer):
        ins, outs = [], []
        for s in SEEDS:
            f = trainer(s)
            ins.append(mse(f(Xin), Yin))
            outs.append(mse(f(Xout), Yout))
        rows.append((name, n_params, np.median(ins), np.median(outs)))

    evaluate('rotor neuron (sandwich)', 3,
             lambda s: train_neuron(Xtr, Ytr, 'sandwich', s))
    evaluate('spinor neuron (one-sided)', 3,
             lambda s: train_neuron(Xtr, Ytr, 'one-sided', s))
    evaluate(f'linear map ({d}×{d})', d * d,
             lambda s: train_linear(Xtr, Ytr, idx))
    for h in WIDTHS:
        evaluate(f'tanh MLP, width {h}', (2 * d + 1) * h + d,
                 lambda s, h=h: train_mlp(Xtr, Ytr, idx, h, s))
    return rows


def run_sample_sweep(kind, sizes=(2, 3, 5, 10, 20, 50, 100), noise=0.05, trials=8):
    """
    Few or noisy samples: matching neuron (3 parameters) vs linear map (d²).

    Targets get Gaussian noise of standard deviation `noise` per component;
    the test set (the held-out wedge) is clean. For each size, `trials`
    independent training sets are drawn and the median test MSE is kept.
    The neuron is started twice per trial and keeps the start with the lower
    training loss, so a bad start is not counted as the model's error.
    """
    idx = VEC if kind == 'vector' else EVEN
    action = 'sandwich' if kind == 'vector' else 'one-sided'
    rng = np.random.default_rng(7)
    Xout, Yout = make_task(kind, 512, (270, 360), rng)

    rows = []
    for n in sizes:
        e_neuron, e_linear = [], []
        for t in range(trials):
            X, Y = make_task(kind, n, (0, 270), rng)
            Y = Y.copy()
            Y[:, idx] += rng.normal(scale=noise, size=(n, len(idx)))
            fits = [train_neuron(X, Y, action, seed=100 * t + k) for k in range(2)]
            best = min(fits, key=lambda f: mse(f(X), Y))
            e_neuron.append(mse(best(Xout), Yout))
            e_linear.append(mse(train_linear(X, Y, idx)(Xout), Yout))
        rows.append((n, np.median(e_neuron), np.median(e_linear)))
    return rows


def print_sweep(title, rows, d):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"  {'samples':>7}{'neuron (3 params)':>20}{f'linear ({d * d} params)':>22}{'ratio':>9}")
    for n, e_n, e_l in rows:
        print(f"  {n:>7}{e_n:>20.2e}{e_l:>22.2e}{e_l / e_n:>9.1f}")


def print_rows(title, rows):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"  {'model':<28}{'params':>7}{'test MSE (seen)':>17}{'test MSE (wedge)':>18}")
    for name, n, e_in, e_out in rows:
        print(f"  {name:<28}{n:>7}{e_in:>17.2e}{e_out:>18.2e}")


def plot(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (title, rows) in zip(axes, results):
        mlp = [r for r in rows if r[0].startswith('tanh')]
        ax.plot([r[1] for r in mlp], [r[2] for r in mlp], 'o-', color='gray',
                label='tanh MLP (seen)')
        ax.plot([r[1] for r in mlp], [r[3] for r in mlp], 'o--', color='black',
                label='tanh MLP (held-out wedge)')
        for r, colour in zip(rows[:3], ['tab:orange', 'tab:blue', 'tab:green']):
            ax.scatter([r[1]], [r[3]], s=90, color=colour, zorder=3,
                       label=f'{r[0]} (wedge)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('parameters')
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('test MSE')
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('rotation_experiment.png', dpi=150, bbox_inches='tight')
    print("\nSaved: rotation_experiment.png")


if __name__ == "__main__":
    import sys
    if '--sweep' in sys.argv:
        print_sweep("FEW, NOISY SAMPLES: vectors (noise 0.05)",
                    run_sample_sweep('vector'), 3)
        print_sweep("FEW, NOISY SAMPLES: spinors (noise 0.05)",
                    run_sample_sweep('spinor'), 4)
        sys.exit()

    a = run_task('vector')
    print_rows("TASK A: learn v → R*·v·R*†  (vectors)", a)
    b = run_task('spinor')
    print_rows("TASK B: learn x → R*·x  (spinors / unit quaternions)", b)
    try:
        plot([("Task A: vectors", a), ("Task B: spinors", b)])
    except ImportError:
        print("\n(matplotlib not installed: skipped the plot)")
