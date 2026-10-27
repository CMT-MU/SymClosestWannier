"""
reference model for the tests of the responses and of tb_gauge: an orthonormal frame V(k) = sum_R e^{ik.R} V_R
(D x W, V^† V = 1; the first W columns of U_0 E_1(k) U_1 E_2(k) ... with unitary U_j and E_j(k) = 1 - P_j + P_j e^{ik.n_j},
P_j a projector) and a Hamiltonian H(k) = V h(k) V^† + Q H_out Q (Q = 1 - V V^†) for which span V(k) is invariant, as
for isolated bands. The operators in the wannier90 convention are, e.g., A_a(k) = i V^† d_a V, BB_a(k) = i V^† H d_a V,
CC_ab(k) = (d_a V)^† H d_b V; all are trigonometric polynomials, so their real-space matrices are obtained exactly by a
discrete Fourier transform on a large enough mesh. In the tb gauge the frame is V^I(k) = V(k) e^{ik.τ}; its operators are
evaluated directly, with derivatives by central finite differences.

This module does not import symclosestwannier.analyzer, so that the tests of the responses can be run against any
implementation of it.
"""

import numpy as np

A = np.array([[2.5, 0.0, 0.0], [-1.25, 2.165, 0.0], [0.3, 0.4, 7.0]])
B = 2 * np.pi * np.linalg.inv(A).T
# orbital positions (fractional), one outside [0, 1).
TAU = np.array([[0.1, 0.2, 0.05], [0.6, 0.3, -0.2], [1.3, 0.7, 0.4]])
NS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
DIM, NUM_WANN = 5, 3
# harmonics of the operators lie in [-3, 3]^3, so a 7^3 mesh with R in [-3, 3]^3 represents them exactly.
NMESH = 7
H_STEP = 1e-5


# ==================================================
class Model:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)

        def herm(n):
            M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            return M + M.conj().T

        def unitary(n):
            return np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]

        def projector(n, rank):
            Q = unitary(n)[:, :rank]
            return Q @ Q.conj().T

        def mul(X, Y):
            Z = {}
            for r1, x in X.items():
                for r2, y in Y.items():
                    r = tuple(np.add(r1, r2))
                    Z[r] = Z.get(r, 0) + x @ y
            return Z

        # U(k) as {R: U_R}
        U = {(0, 0, 0): unitary(DIM)}
        for n in NS:
            P = projector(DIM, 2)
            U = mul(U, {(0, 0, 0): np.eye(DIM) - P, n: P})
            U = mul(U, {(0, 0, 0): unitary(DIM)})

        self.NV = np.array(list(U.keys()))
        self.V_R = np.array([U[n][:, :NUM_WANN] for n in U])
        self.nv_cart = self.NV @ A

        # h(k) = h_0 + sum_j (h_j e^{ik.n_j} + h.c.)
        h1 = rng.normal(size=(NUM_WANN, NUM_WANN)) + 1j * rng.normal(size=(NUM_WANN, NUM_WANN))
        self.h_R = {(0, 0, 0): herm(NUM_WANN), (1, 0, 0): h1, (-1, 0, 0): h1.conj().T}
        self.H_out = herm(DIM)
        self.sigma = np.array([herm(DIM) for _ in range(3)])

        irvec = [(i, j, l) for i in range(-3, 4) for j in range(-3, 4) for l in range(-3, 4)]
        self.irvec = np.array(irvec)
        self.ndegen = np.ones(len(irvec), dtype=int)

    # k: fractional coordinates.
    def V(self, k):
        return np.einsum("r,rdw->dw", np.exp(2j * np.pi * self.NV @ k), self.V_R)

    def dV(self, k):
        """analytic cartesian derivative, (3, D, W)."""
        return np.einsum("r,ra,rdw->adw", np.exp(2j * np.pi * self.NV @ k), 1j * self.nv_cart, self.V_R)

    def VI(self, k):
        return self.V(k) * np.exp(2j * np.pi * TAU @ k)[None, :]

    def dVI(self, k):
        """finite-difference cartesian derivative of the tb-gauge frame, (3, D, W)."""
        dk = H_STEP * np.linalg.inv(B)  # rows: fractional steps along cartesian x, y, z
        return np.array([(self.VI(k + dk[a]) - self.VI(k - dk[a])) / (2 * H_STEP) for a in range(3)])

    def H(self, k):
        """Hamiltonian in the full space, span V(k) is invariant."""
        V = self.V(k)
        h = sum(np.exp(2j * np.pi * np.dot(k, R)) * hR for R, hR in self.h_R.items())
        Q = np.eye(DIM) - V @ V.conj().T
        return V @ h @ V.conj().T + Q @ self.H_out @ Q

    # ==================================================
    def operators_k(self, k, V, dV):
        H, S = self.H(k), self.sigma
        Vd = V.conj().T
        return {
            "HH": Vd @ H @ V,
            "AA": np.array([1j * Vd @ dV[a] for a in range(3)]),
            "BB": np.array([1j * Vd @ H @ dV[a] for a in range(3)]),
            "CC": np.array([[dV[a].conj().T @ H @ dV[b] for b in range(3)] for a in range(3)]),
            "SS": np.array([Vd @ S[a] @ V for a in range(3)]),
            "SR": np.array([[1j * Vd @ S[a] @ dV[b] for b in range(3)] for a in range(3)]),
            "SH": np.array([Vd @ S[a] @ H @ V for a in range(3)]),
            "SHR": np.array([[1j * Vd @ S[a] @ H @ dV[b] for b in range(3)] for a in range(3)]),
        }

    def operators_R(self):
        """
        wannier90-convention operators, O(R) = 1/N_k sum_k e^{-ik.R} O(k), not divided by ndegen (= 1).
        """
        if not hasattr(self, "_operators_R"):
            kpoints = np.array([[i, j, l] for i in range(NMESH) for j in range(NMESH) for l in range(NMESH)]) / NMESH
            ops = [self.operators_k(k, self.V(k), self.dV(k)) for k in kpoints]
            phase = np.exp(-2j * np.pi * kpoints @ self.irvec.T) / len(kpoints)
            d = {}
            for key in ops[0]:
                Ok = np.array([o[key] for o in ops])  # (k, ..., m, n)
                Ok = np.moveaxis(Ok, 0, -3)  # (..., k, m, n)
                d[f"{key}_R"] = np.einsum("kR,...kmn->...Rmn", phase, Ok)
            self._operators_R = d

        return {k: v.copy() for k, v in self._operators_R.items()}


# ==================================================
def make_cwi(model, tb_gauge):
    return {
        "tb_gauge": tb_gauge,
        # an atom without projection comes first, and the projection centres are listed in another order than the
        # orbitals: the orbital positions must come from atom_pos_r[nw2n], not from atoms_frac.
        "atoms_frac": {("O", 1): [0.5, 0.5, 0.5], **{("X", i + 1): list(t) for i, t in enumerate(TAU)}},
        "atom_pos_r": [list(TAU[2]), list(TAU[0]), list(TAU[1])],
        "nw2n": [1, 2, 0],
        "num_wann": NUM_WANN,
        "num_fermi": 1,
        "fermi_energy_list": [0.0],
        "unit_cell_cart": A,
        "irvec": model.irvec,
        "ndegen": model.ndegen,
        "zeeman_interaction": False,
        "shc_alpha": 1,
        "shc_gamma": 3,
    }
