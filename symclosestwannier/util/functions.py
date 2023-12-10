"""
utility functions.
"""

import numpy as np
from gcoreutils.nsarray import NSArray


M_ZERO = np.finfo(float).eps


# ==================================================
def is_zero(x):
    return np.abs(x) < M_ZERO * 100


# ==================================================
def fermi(x, T=0.01):
    return 0.5 * (1.0 - np.tanh(0.5 * x / T))


# ==================================================
def fermi_dt(x, T=0.01):
    return fermi(x, T) * fermi(-x, T) / T


# ==================================================
def fermi_ddt(x, T=0.01):
    return (1 - 2 * fermi(-x, T)) * fermi(x, T) * fermi(-x, T) / T / T


# ==================================================
def w_proj(e, e0, e1, T0, T1, delta=10e-12):
    """weight function for projection"""
    return fermi(e0 - e, T0) + fermi(e - e1, T1) - 1.0 + delta


# ==================================================
def get_rpoints(nr1, nr2, nr3, unit_cell_cart=np.eye(3)):
    """
    get lattice points, R = (R1, R2, R3).
    R = R1*a1 + R2*a2 + R3*a3
    Rj: lattice vector.

    Args:
        nr1 (int): # of lattice point a1 direction.
        nr2 (int): # of lattice point a2 direction.
        nr3 (int): # of lattice point a3 direction.
        unit_cell_cart (ndarray, optional): transform matrix, [a1,a2,a3].

    Returns:
        tuple: (R, Rfft, idx).
    """
    A = unit_cell_cart
    nrtot = nr1 * nr2 * nr3

    R = np.zeros((nrtot, 3), dtype=float)
    idx = np.zeros((nr1, nr2, nr3), dtype=int)
    Rfft = np.zeros((nr1, nr2, nr3, 3), dtype=float)

    for i in range(nr1):
        for j in range(nr2):
            for k in range(nr3):
                n = k + j * nr3 + i * nr2 * nr3
                R1 = float(i) / float(nr1)
                R2 = float(j) / float(nr2)
                R3 = float(k) / float(nr3)
                if R1 >= 0.5:
                    R1 = R1 - 1.0
                if R2 >= 0.5:
                    R2 = R2 - 1.0
                if R3 >= 0.5:
                    R3 = R3 - 1.0
                R1 -= int(R1)
                R2 -= int(R2)
                R3 -= int(R3)

                R[n, :] = R1 * nr1 * A[0, :] + R2 * nr2 * A[1, :] + R3 * nr3 * A[2, :]
                Rfft[i, j, k, :] = R[n, :]
                idx[i, j, k] = n

    return R, Rfft, idx


# ==================================================
def get_kpoints(nk1, nk2, nk3):
    """
    get k-points (crystal coordinate), k = (k1, k2, k3).
    k = k1*b1 + k2*b2 + k3*b3
    bj: reciprocal lattice vector.

    Args:
        nk1 (int): # of lattice point b1 direction.
        nk2 (int): # of lattice point b2 direction.
        nk3 (int): # of lattice point b3 direction.

    Returns:
        ndarray: lattice points.
    """
    nktot = nk1 * nk2 * nk3

    Kint = np.zeros((nktot, 3), dtype=float)

    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                n = k + j * nk3 + i * nk2 * nk3
                k1 = float(i) / float(nk1)
                k2 = float(j) / float(nk2)
                k3 = float(k) / float(nk3)
                if k1 >= 0.5:
                    k1 = k1 - 1.0
                if k2 >= 0.5:
                    k2 = k2 - 1.0
                if k3 >= 0.5:
                    k3 = k3 - 1.0
                k1 -= int(k1)
                k2 -= int(k2)
                k3 -= int(k3)

                Kint[n] = k1, k2, k3

    return Kint


# ==================================================
def kpoints_to_rpoints(kpoints):
    """
    get lattice points corresponding to k-points.

    Args:
        kpoints (ndarray): k-points (crystal coordinate).
        nk3 (int): # of lattice point b3 direction.

    Returns:
        ndarray: k-points (crystal coordinate).
    """
    kpoints = np.array(kpoints, dtype=float)
    N1 = len(sorted(set(list(kpoints[:, 0]))))
    N2 = len(sorted(set(list(kpoints[:, 1]))))
    N3 = len(sorted(set(list(kpoints[:, 2]))))
    N1 = N1 - 1 if N1 % 2 == 0 else N1
    N2 = N2 - 1 if N2 % 2 == 0 else N2
    N3 = N3 - 1 if N3 % 2 == 0 else N3
    rpoints, _, _ = get_rpoints(N1, N2, N3)

    return rpoints


# ==================================================
def fourier_transform_k_to_r(Ok, kpoints, rpoints=None, atoms_frac=None):
    """
    inverse fourier transformation of an arbitrary operator from k-space representation into real-space representation.

    Args:
        Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
        kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
        rpoints (ndarray, optional): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
        atoms_frac (ndarray, optional): atom's position in fractional coordinates.

    Returns:
        (ndarray, ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>, lattice points.
    """
    # lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
    if rpoints is None:
        rpoints = kpoints_to_rpoints(kpoints)

    Ok = np.array(Ok, dtype=complex)
    kpoints = np.array(kpoints, dtype=float)
    rpoints = np.array(rpoints, dtype=float)

    # number of k points
    num_k = kpoints.shape[0]
    # number of lattice points
    Nr = rpoints.shape[0]
    # number of pseudo atomic orbitalså
    num_wann = Ok.shape[1]

    if atoms_frac is not None:
        ap = np.array(atoms_frac)
        phase_ab = np.exp(
            [
                [1.0j * (2 * np.pi * kpoints @ (ap[a, :] - ap[b, :]).transpose()) for b in range(num_wann)]
                for a in range(num_wann)
            ]
        ).transpose(2, 0, 1)
        Ok = Ok * phase_ab

    phase = np.exp(1.0j * 2 * np.pi * kpoints @ rpoints.T)
    Or = np.array([np.sum(Ok[:, :, :] * phase[:, r, np.newaxis, np.newaxis], axis=0) for r in range(Nr)])
    Or /= num_k

    rpoints = np.array([[round(N1), round(N2), round(N3)] for N1, N2, N3 in rpoints], dtype=int)

    return Or, rpoints


# ==================================================
def fourier_transform_r_to_k(Or, rpoints, kpoints, atoms_frac=None):
    """
    fourier transformation of an arbitrary operator from real-space representation into k-space representation.

    Args:
        Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>.
        rpoints (ndarray): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
        kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
        atoms_frac (ndarray, optional): atom's position in fractional coordinates.

    Returns:
        ndarray: k-space representation of the given operator, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
    """
    Or = np.array(Or, dtype=complex)
    rpoints = np.array(rpoints, dtype=float)
    kpoints = np.array(kpoints, dtype=float)

    # number of k points
    num_k = kpoints.shape[0]

    # number of pseudo atomic orbitalså
    num_wann = Or.shape[1]

    phase = np.exp(-1.0j * 2 * np.pi * kpoints @ rpoints.T)
    Ok = np.array([np.sum(Or[:, :, :] * phase[k, :, np.newaxis, np.newaxis], axis=0) for k in range(num_k)])

    if atoms_frac is not None:
        ap = np.array(atoms_frac)
        phase_ab = np.exp(
            [
                [1.0j * (-2 * np.pi * kpoints @ (ap[a, :] - ap[b, :]).transpose()) for b in range(num_wann)]
                for a in range(num_wann)
            ]
        ).transpose(2, 0, 1)
        Ok = Ok * phase_ab

    return Ok, kpoints


# ==================================================
def interpolate(Ok, kpoints_0, kpoints, rpoints=None, atoms_frac=None):
    """
    interpolate an arbitrary operator by implementing
    fourier transformation from real-space representation into k-space representation.

    Args:
        Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
        kpoints_0 (ndarray): k points before interpolated (crystal coordinate, [[k1,k2,k3]]).
        kpoints (ndarray): k points after interpolated (crystal coordinate, [[k1,k2,k3]]).
        rpoints (ndarray): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
        atoms_frac (ndarray, optional): atom's position in fractional coordinates.

    Returns:
        ndarray: matrix elements at each k point, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
    """
    Or, rpoints = fourier_transform_k_to_r(Ok, kpoints_0, rpoints, atoms_frac)
    Ok_interpolated, _ = fourier_transform_r_to_k(Or, rpoints, kpoints, atoms_frac)

    return Ok_interpolated


# ==================================================
def matrix_dict_r(Or, rpoints, diagonal=False):
    """
    dictionary form of an arbitrary operator matrix in real-space representation.

    Args:
        Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>.
        rpoints (ndarray): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
        diagonal (bool, optional): diagonal matrix ?

    Returns:
        dict: real-space representation of the given operator, {(n2,n2,n3,a,b) = O_{ab}(R)}.
    """
    # number of pseudo atomic orbitals
    dim_r = len(Or[0])
    if not diagonal:
        dim_c = len(Or[0][0])

    Or_dict = {}

    r_list = [[r, round(n1), round(n2), round(n3)] for r, (n1, n2, n3) in enumerate(rpoints)]

    if diagonal:
        Or_dict = {(n1, n2, n3, a, a): Or[r][a] for r, n1, n2, n3 in r_list for a in range(dim_r)}
    else:
        Or_dict = {
            (n1, n2, n3, a, b): Or[r][a][b] for r, n1, n2, n3 in r_list for a in range(dim_r) for b in range(dim_c)
        }

    return Or_dict


# ==================================================
def matrix_dict_k(Ok, kpoints, diagonal=False):
    """
    dictionary form of an arbitrary operator matrix in k-space representation.

    Args:
        Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|H|φ_{b}(k)>.
        kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
        diagonal (bool, optional): diagonal matrix ?

    Returns:
        dict: k-space representation of the given operator, {(k2,k2,k3,a,b) = O_{ab}(k)}.
    """
    # number of pseudo atomic orbitals
    dim_r = len(Ok[0])
    if not diagonal:
        dim_c = len(Ok[0][0])

    k_list = [[k, k1, k2, k3] for k, (k1, k2, k3) in enumerate(kpoints)]

    if diagonal:
        Ok_dict = {(k1, k2, k3, a, a): Ok[k][a] for k, k1, k2, k3 in k_list for a in range(dim_r)}
    else:
        Ok_dict = {
            (k1, k2, k3, a, b): Ok[k][a][b] for k, k1, k2, k3 in k_list for a in range(dim_r) for b in range(dim_c)
        }

    return Ok_dict


# ==================================================
def dict_to_matrix(Or_dict):
    """
    convert dictionary form to matrix form of an arbitrary operator matrix.

    Args:
        Or_dict (dict): dictionary form of an arbitrary operator matrix in reak-space/k-space representation.

    Returns:
        ndarray: matrix form of the given operator.
    """
    dim = max([a for (_, _, _, a, _) in Or_dict.keys()]) + 1

    O_mat = [np.zeros((dim, dim), dtype=complex)]
    idx = 0
    g0 = list(Or_dict.keys())[0][:3]

    for (g1, g2, g3, a, b), v in Or_dict.items():
        g = (g1, g2, g3)
        if g != g0:
            O_mat.append(np.zeros((dim, dim), dtype=complex))
            idx += 1
            g0 = g

        O_mat[idx][a, b] = complex(v)

    return np.array(O_mat)


# ==================================================
def samb_decomp(Or_dict, Zr_dict):
    """
    decompose arbitrary operator into linear combination of SAMBs.

    Args:
        Or_dict (dict): dictionary form of an arbitrary operator matrix in reak-space/k-space representation.
        Zr_dict (dict): SAMBs

    Returns:
        z (list): parameter set, [z_j].
    """
    z = {
        k: np.real(np.sum([v * Or_dict.get((-k[0], -k[1], -k[2], k[4], k[3]), 0) for k, v in d.items()]))
        for k, d in Zr_dict.items()
    }

    return z


# ==================================================
def construct_Or(z, num_wann, rpoints, matrix_dict):
    """
    arbitrary operator constructed by linear combination of SAMBs in real space representation.

    Args:
        z (list): parameter set, [z_j].
        num_wann (int): # of CWFs.
        rpoints (ndarray, optional): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
        matrix_dict (dict): SAMBs.

    Returns:
        ndarray: matrix, [#r, dim, dim].
    """
    Or_dict = {(n1, n2, n3, a, b): 0.0 for (n1, n2, n3) in rpoints for a in range(num_wann) for b in range(num_wann)}
    for j, d in enumerate(matrix_dict["matrix"].values()):
        zj = z[j]
        for (n1, n2, n3, a, b), v in d.items():
            Or_dict[(n1, n2, n3, a, b)] += zj * v

    Or = np.array(
        [
            [[Or_dict.get((n1, n2, n3, a, b), 0.0) for b in range(num_wann)] for a in range(num_wann)]
            for (n1, n2, n3) in rpoints
        ]
    )

    return Or


# ==================================================
def construct_Ok(z, num_wann, kpoints, rpoints, matrix_dict):
    """
    arbitrary operator constructed by linear combination of SAMBs in k space representation.

    Args:
        z (list): parameter set, [z_j].
        num_wann (int): # of CWFs.
        kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
        rpoints (ndarray, optional): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
        matrix_dict (dict): SAMBs.

    Returns:
        ndarray: matrix, [#k, dim, dim].
    """
    kpoints = np.array(kpoints)
    rpoints = np.array(rpoints)
    cell_site = matrix_dict["cell_site"]
    ket = matrix_dict["ket"]
    atoms_frac = [
        NSArray(cell_site[ket[a].split("@")[1]][0], style="vector", fmt="value").tolist() for a in range(len(ket))
    ]

    Or = construct_Or(z, num_wann, rpoints, matrix_dict)
    Ok, _ = fourier_transform_r_to_k(Or, rpoints, kpoints, atoms_frac)

    return Ok
