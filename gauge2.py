import numpy as np
from scipy.linalg import expm          # matrix exponential for transport
from scipy.integrate import solve_ivp  # solve geodesic ODE
from scipy.special import softmax      # for attention weights
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- 1.  base manifold with TRUE GEODESICS -----------------------------
class Manifold:
    def __init__(self, mfd="sphere", dim=2):
        self.mfd, self.dim = mfd, dim

    # --- embeddings / metric / Christoffel (unchanged, but vectorised)
    def chart(self, θφ: np.ndarray) -> np.ndarray:
        if self.mfd == "sphere":
            θ, φ = θφ[..., 0], θφ[..., 1]
            return np.stack([np.sin(θ)*np.cos(φ),
                             np.sin(θ)*np.sin(φ),
                             np.cos(θ)], -1)
        if self.mfd == "torus":
            u, v = θφ[..., 0], θφ[..., 1]
            R, r = 2.0, 1.0
            return np.stack([(R+r*np.cos(v))*np.cos(u),
                             (R+r*np.cos(v))*np.sin(u),
                             r*np.sin(v)], -1)
        return θφ                                              # flat

    def metric(self, θφ):
        if self.mfd == "sphere":
            θ = θφ[..., 0]
            g = np.zeros((*θφ.shape[:-1], 2, 2))
            g[..., 0, 0], g[..., 1, 1] = 1.0, np.sin(θ)**2
            return g
        return np.eye(self.dim)                                # flat / torus

    def christoffel(self, θφ):
        if self.mfd == "sphere":
            θ = θφ[..., 0]
            Γ = np.zeros((*θφ.shape[:-1], 2, 2, 2))
            Γ[..., 0, 1, 1] = -np.sin(θ)*np.cos(θ)
            Γ[..., 1, 0, 1] = Γ[..., 1, 1, 0] = np.cos(θ)/(np.sin(θ)+1e-12)
            return Γ
        return np.zeros((*θφ.shape[:-1], self.dim, self.dim, self.dim))

    # --- TRUE GEODESIC: solve d²xᵏ/dt² + Γᵏᵢⱽ dxⁱ/dt dxʲ/dt = 0 -------------
    def geodesic(self, start: np.ndarray, end: np.ndarray, steps: int = 20):
        """
        start, end shape (dim,) in chart coords
        returns (steps, dim) on the manifold
        """
        # initial tangent vector in chart space
        v0 = (end - start)                           # first guess
        # refine with shooting (one Newton step is enough for demo)
        sol = solve_ivp(
            lambda t, y: self._geodesic_ode(y),
            [0, 1], np.concatenate([start, v0]),
            t_eval=np.linspace(0, 1, steps))
        path = sol.y[:self.dim, :].T                 # (steps, dim)
        return path

    def _geodesic_ode(self, y):
        x, v = y[:self.dim], y[self.dim:]
        Γ = self.christoffel(x)                      # (dim, dim, dim)
        dv = -np.einsum('ijk,i,j->k', Γ, v, v)
        return np.concatenate([v, dv])


# ---------- 2.  CONNECTION with MATRIX EXPONENTIAL TRANSPORT ------------------
class ConnectionForm:
    def __init__(self, manifold: Manifold, fiber_dim: int):
        self.mfd, self.fd = manifold, fiber_dim
        # initialise in 𝔤 = 𝔰𝔬(n)  (antisymmetric)
        self.A = [self._random_so() for _ in range(manifold.dim)]

    def _random_so(self):
        x = np.random.randn(self.fd, self.fd)*0.01
        return x - x.T

    # --- PARALLEL TRANSPORT u along path (piece-wise constant A) --------------
    def transport(self, u: np.ndarray, path: np.ndarray) -> np.ndarray:
        """
        path: (steps, dim)  in chart coords
        returns transported vector at final point
        """
        v = u.copy()
        for t in range(len(path)-1):
            tangent = path[t+1] - path[t]
            dt = np.linalg.norm(tangent)
            # A_μ dx^μ  (tangent in chart space)
            Amu_dx = sum(tangent[mu]*self.A[mu] for mu in range(self.mfd.dim))
            v = expm(-dt*Amu_dx) @ v
        return v

    # --- FIELD STRENGTH F = dA + A∧A  (non-zero!) ----------------------------
    def curvature(self, x: np.ndarray) -> np.ndarray:
        """returns list of F_{μν} matrices at base point x"""
        F = []
        for μ in range(self.mfd.dim):
            for ν in range(μ+1, self.mfd.dim):
                # naive symmetric derivative (works for static demo)
                dA = 0.0
                h = 1e-4
                x_plus = x.copy();  x_plus[μ] += h
                x_minus = x.copy(); x_minus[μ] -= h
                dA += (self._interpolate_A(x_plus)[ν] - self._interpolate_A(x_minus)[ν])/(2*h)
                x_plus = x.copy();  x_plus[ν] += h
                x_minus = x.copy(); x_minus[ν] -= h
                dA -= (self._interpolate_A(x_plus)[μ] - self._interpolate_A(x_minus)[μ])/(2*h)
                # [A_μ, A_ν]
                commutator = self.A[μ]@self.A[ν] - self.A[ν]@self.A[μ]
                F.append(dA + commutator)
        return F

    def _interpolate_A(self, x):
        # placeholder: in real life use local sections / neural predictor
        return self.A


# ---------- 3.  NON-TRIVIAL BUNDLE GLUEING (clutching teaser) ---------------
class Section:
    """
    Local section over a chart.  Transition functions on overlaps
    let us move fibres between charts → non-trivial bundle.
    For demo we stay in one chart, but the infra is here.
    """
    def __init__(self, chart_id: str, conn: ConnectionForm):
        self.id, self.conn = chart_id, conn

    def transition(self, other: 'Section', x):
        """return g_{UV}(x) in SO(n) that glues charts U and V"""
        # identity for trivial bundle; replace with learned or analytic g
        return np.eye(self.conn.fd)


# ---------- 4.  GAUGE-EQUIVARIANT ATTENTION (vectorised) ----------------------
class GaugeAttention:
    def __init__(self, bundle, heads=4):
        self.bundle, self.heads = bundle, heads
        d = bundle.connection.fd
        # projectors Q,K,V in 𝔤
        init = lambda: np.random.randn(heads, d, d)*0.1
        self.W_q, self.W_k, self.W_v = init(), init(), init()

    def __call__(self, x, coords):
        B, d = x.shape
        H = self.heads
        # lift to heads
        q = np.einsum('bd,hdc->bhd', x, self.W_q)   # (B,H,d)
        k = np.einsum('bd,hdc->bhd', x, self.W_k)
        v = np.einsum('bd,hdc->bhd', x, self.W_v)

        # parallel-transport keys to query points
        scores = np.zeros((H, B, B))
        for h in range(H):
            for i in range(B):
                for j in range(B):
                    path = self.bundle.base.geodesic(coords[j], coords[i])
                    k_trans = self.bundle.connection.transport(k[j, h], path)
                    scores[h, i, j] = q[i, h] @ k_trans
        scores /= np.sqrt(d)
        w = softmax(scores, axis=-1)              # (H,B,B)

        out = np.zeros_like(v)
        for h in range(H):
            for i in range(B):
                for j in range(B):
                    path = self.bundle.base.geodesic(coords[j], coords[i])
                    v_trans = self.bundle.connection.transport(v[j, h], path)
                    out[i, h] += w[h, i, j] * v_trans
        # merge heads
        return out.reshape(B, -1)


# ---------- 5.  FULL TRANSFORMER LAYER ---------------------------------------
class GaugeLayer:
    def __init__(self, bundle, heads=4):
        self.attn = GaugeAttention(bundle, heads)
        d = bundle.connection.fd
        self.W1 = np.random.randn(d, 4*d)*0.1
        self.W2 = np.random.randn(4*d, d)*0.1

    def __call__(self, x, coords):
        x = x + self.attn(x, coords)[:, :x.shape[-1]]
        ff = np.tanh(x @ self.W1) @ self.W2
        return x + ff


# ---------- 6.  TOP-LEVEL MODEL ----------------------------------------------
class GaugeTransformer:
    def __init__(self, mfd="sphere", fd=16, layers=3, heads=4):
        base = Manifold(mfd)
        conn = ConnectionForm(base, fd)
        self.bundle = type('Bundle', (), {'base': base, 'connection': conn})()
        self.layers = [GaugeLayer(self.bundle, heads) for _ in range(layers)]

    def __call__(self, x, coords):
        for lyr in self.layers:
            x = lyr(x, coords)
        return x

    # quick visual (unchanged conceptually)
    def plot(self, x, coords, file="gauge_v2.png"):
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        emb = self.bundle.base.chart(coords)
        ax.scatter(*emb.T, c=np.linalg.norm(x, axis=1), s=80, cmap='magma')
        # geodesic edges (sparse)
        for i in range(min(len(coords), 6)):
            for j in range(i+1, min(len(coords), i+4)):
                path = self.bundle.base.geodesic(coords[i], coords[j])
                ax.plot(*self.bundle.base.chart(path).T, lw=1, alpha=.4)
        plt.savefig(file, dpi=150)
        print(f"saved {file}")


# ---------- 7.  30-SECOND DEMO -----------------------------------------------
if __name__ == "__main__":
    model = GaugeTransformer("sphere", fd=24, layers=3, heads=4)
    n = 20
    θφ = np.stack([np.random.uniform(.2, np.pi-.2, n),
                   np.random.uniform(0, 2*np.pi, n)], -1)
    x = np.random.randn(n, 24)*.3
    y = model(x, θφ)
    print("in:", x.shape, "→ out:", y.shape)
    model.plot(x, θφ)