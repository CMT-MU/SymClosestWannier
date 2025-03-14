"""
utility codes.
"""

import datetime
import itertools
import textwrap
import fortio
import numpy as np
from sympy.physics.quantum import TensorProduct
from heapq import nlargest
import multiprocessing
from joblib import Parallel, delayed, wrap_non_picklable_objects

from gcoreutils.nsarray import NSArray

from symclosestwannier.util.constants import k_B_SI, elem_charge_SI, bohr_magn_SI, joul_to_eV

M_ZERO = np.finfo(float).eps

_num_proc = multiprocessing.cpu_count()


# ==================================================
class FortranFileR(fortio.FortranFile):
    def __init__(self, filename):
        try:
            super().__init__(filename, mode="r", header_dtype="uint32", auto_endian=True, check_file=True)
        except ValueError:
            print("File '{}' contains subrecords - using header_dtype='int32'".format(filename))
            super().__init__(filename, mode="r", header_dtype="int32", auto_endian=True, check_file=True)


# ==================================================
def is_zero(x):
    return np.abs(x) < M_ZERO * 100


# ==================================================
def Kelvin_to_eV(T_Kelvin):
    """convert K to eV"""
    return T_Kelvin * k_B_SI / elem_charge_SI


# ==================================================
def fermi(x, T=0.0, unit="Kelvin"):
    T_eV = Kelvin_to_eV(T) if unit == "Kelvin" else T

    if T == 0.0:
        return 1.0 * np.vectorize(float)(x < 0.0)

    return 0.5 * (1.0 - np.tanh(0.5 * x / T_eV))


# ==================================================
def fermi_dt(x, T=0.01, unit="Kelvin"):
    T_eV = Kelvin_to_eV(T) if unit == "Kelvin" else T
    return fermi(x, T, unit) * fermi(-x, T, unit) / T_eV


# ==================================================
def fermi_ddt(x, T=0.01, unit="Kelvin"):
    T_eV = Kelvin_to_eV(T) if unit == "Kelvin" else T
    return (1 - 2 * fermi(-x, T, unit)) * fermi(x, T, unit) * fermi(-x, T, unit) / T_eV / T_eV


# ==================================================
def weight_proj(e, dis_win_emin, dis_win_emax, smearing_temp_min, smearing_temp_max, delta=10e-12):
    """weight function for projection"""
    return (
        fermi(dis_win_emin - e, smearing_temp_min, unit="eV")
        + fermi(e - dis_win_emax, smearing_temp_max, unit="eV")
        - 1.0
        + delta
    )


# ==================================================
def band_distance(Ak, Ek, Hk, ef=0.0):
    """
    band distance defined in [npj Computational Materials, 208 (2023)]:
        - eta_x = \sqrt{ \sum w_{nk}(x) (e_{nk}^{DFT} - e_{nk}^{Wan})**2 / \sum w_{nk} }.
        - eta_x_max = \max{ w_{nk}(x) \abs{e_{nk}^{DFT} - e_{nk}^{Wan}} }.
        - w_{nk}(x) = \sqrt{ f_{nk}^{DFT}(Ef+x,T=0.1) * f_{nk}^{Wan}(Ef+x,T=0.1) }
        - f_{nk}: fermi-dirac distribution function.

    Args:
        Ak (ndarray): Overlap matrix elements.
        Ek (ndarray): Kohn-Sham energies.
        Hk (ndarray): Hamiltonian matrix elements in k-space (orthogonal).
        ef (float, optional): fermi energy.

    Returns: eta_0, eta_0_max, eta_2, eta_2_max.
    """
    num_k, num_wann, _ = Hk.shape

    Ek_wan, _ = np.linalg.eigh(Hk)

    # projectability of each Kohn-Sham state in k-space.
    Pk = np.real(np.diagonal(Ak @ Ak.transpose(0, 2, 1).conjugate(), axis1=1, axis2=2))

    Ek_ref = np.zeros((num_k, num_wann))
    for k in range(num_k):
        Pk_ = [(pnk, n) for n, pnk in enumerate(Pk[k])]
        Pk_max_idx_list = sorted([n for _, n in nlargest(num_wann, Pk_)])
        for i, n in enumerate(Pk_max_idx_list):
            Ek_ref[k, i] = Ek[k, n]

    for k in range(num_k):
        for n in range(num_wann):
            if Ek_ref[k, n] > ef or Ek_wan[k, n] > ef:
                Ek_ref[k, n] = 0.0
                Ek_wan[k, n] = 0.0

    fermi_ref = fermi(Ek_ref - (ef + 0.0), T=0.1, unit="eV")
    fermi_wan = fermi(Ek_wan - (ef + 0.0), T=0.1, unit="eV")
    w = np.sqrt(fermi_ref * fermi_wan)
    eta_0 = np.sqrt(np.sum(w * (Ek_ref - Ek_wan) ** 2) / np.sum(w)) * 1000
    eta_0_max = np.max(w * np.abs(Ek_ref - Ek_wan)) * 1000

    fermi_ref = fermi(Ek_ref - (ef + 2.0), T=0.1, unit="eV")
    fermi_wan = fermi(Ek_wan - (ef + 2.0), T=0.1, unit="eV")
    w = np.sqrt(fermi_ref * fermi_wan)
    eta_2 = np.sqrt(np.sum(w * (Ek_ref - Ek_wan) ** 2) / np.sum(w)) * 1000
    eta_2_max = np.max(w * np.abs(Ek_ref - Ek_wan)) * 1000

    return eta_0, eta_0_max, eta_2, eta_2_max


# ==================================================
def convert_w90_orbital(l, m, r, s):
    """
    convert orbital in the Wannier90 format into the MultiPie format.

    Args:
        nw2l (list): l specifies the angular part Θlm(θ, φ).
        nw2m (list): m specifies the angular part Θlm(θ, φ).
        nw2r (list): r specifies the radial part Rr(r).
        nw2s (list): s specifies the spin, 1(up)/-1(dn).

    Returns:
        str: converted orbital.
    """
    orbital = ""

    if l == 0 and m == 1:
        orbital = "s"
    elif l == 1:
        if m == 1:
            orbital = "pz"
        elif m == 2:
            orbital = "px"
        elif m == 3:
            orbital = "py"
    elif l == 2:
        if m == 1:
            orbital = "du"  # dz2
        elif m == 2:
            orbital = "dzx"  # dxz
        elif m == 3:
            orbital = "dyz"
        elif m == 4:
            orbital = "dv"  # dx2-y2
        elif m == 5:
            orbital = "dxy"
    elif l == 3:
        if m == 1:
            orbital = "faz"  # fz3
        elif m == 2:
            orbital = "fx"  # fxz2
        elif m == 3:
            orbital = "fy"  # fyz2
        elif m == 4:
            orbital = "fbz"  # fz(x2-y2)
        elif m == 5:
            orbital = "fxyz"
        elif m == 6:
            orbital = "f2"  # fx(x2-3y2)
        elif m == 7:
            orbital = "f1"  # fy(3x2-y2)

    if orbital == "":
        raise Exception(f"invalid orbital projection was given, (l={l},m={m},r={r},s={s}).")

    if s == 1:
        orbital = f"({orbital},U)".replace("'", "")
    elif s == -1:
        orbital = f"({orbital},D)".replace("'", "")

    return orbital


# ==================================================
def iterate_nd(size, pm=False):
    a = -size[0] if pm else 0
    b = size[0] + 1 if pm else size[0]
    if len(size) == 1:
        return np.array([(i,) for i in range(a, b)])
    else:
        return np.array([(i,) + tuple(j) for i in range(a, b) for j in iterate_nd(size[1:], pm=pm)])


# ==================================================
def iterate3dpm(size):
    assert len(size) == 3
    return iterate_nd(size, pm=True)


# ==================================================
def wigner_seitz(A, mp_grid, prec=1.0e-4):
    """
    wigner seitz cell.
    return irreducible R vectors and number of degeneracy at each R.

    Args:
        A (list/ndarray): real lattice vectors, A = [a1,a2,a3] (list), [[[1,0,0], [0,1,0], [0,0,1]]].
        mp_grid (list/ndarray): dimensions of the Monkhorst-Pack grid of k-points (list), [0, 0, 0].,

    Returns:
        tuple: (irvec, ndegen)
            - irvec (ndarray): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
            - ndegen (ndarray): number of degeneracy at each R.
    """
    ws_search_size = np.array([1] * 3)
    dist_dim = np.prod((ws_search_size + 1) * 2 + 1)
    origin = divmod((dist_dim + 1), 2)[0] - 1
    real_metric = A.dot(A.T)
    mp_grid = np.array(mp_grid)
    irvec = []
    ndegen = []
    for n in iterate3dpm(mp_grid * ws_search_size):
        dist = []
        for i in iterate3dpm((1, 1, 1) + ws_search_size):
            ndiff = n - i * mp_grid
            dist.append(ndiff.dot(real_metric.dot(ndiff)))
        dist_min = np.min(dist)
        if abs(dist[origin] - dist_min) < prec:
            irvec.append(n)
            ndegen.append(np.sum(abs(dist - dist_min) < prec))

    ndegen = np.array(ndegen, dtype="int64")
    irvec = np.array(irvec, dtype="int64")

    return irvec, ndegen


# ==================================================
def fourier_transform_k_to_r(Ok, kpoints, irvec, atoms_frac=None):
    """
    inverse fourier transformation of an arbitrary operator from k-space representation into real-space representation.

    Args:
        Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
        kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
        irvec (ndarray, optional): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
        atoms_frac (ndarray, optional): atom's position in fractional coordinates.

    Returns:
        (ndarray, ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(0)|O|φ_{b}(R)>, lattice points.
    """
    Ok = np.array(Ok, dtype=complex)
    kpoints = np.array(kpoints, dtype=float)
    irvec = np.array(irvec, dtype=float)

    num_k = kpoints.shape[0]

    kR = np.einsum("ka,Ra->kR", kpoints, irvec, optimize=True)
    phase_R = np.exp(-2 * np.pi * 1j * kR)

    if atoms_frac is not None:
        tau = np.array(atoms_frac)
        ktau = np.einsum("ka,ma->km", kpoints, tau, optimize=True)
        eiktau = np.exp(+2 * np.pi * 1j * ktau)

        Or = np.einsum("kR,km,kmn,kn->Rmn", phase_R, eiktau, Ok, eiktau.conjugate(), optimize=True) / num_k
    else:
        Or = np.einsum("kR,kmn->Rmn", phase_R, Ok, optimize=True) / num_k

    return Or


# ==================================================
def fourier_transform_r_to_k(Or, kpoints, irvec, ndegen=None, atoms_frac=None):
    """
    fourier transformation of an arbitrary operator from real-space representation into k-space representation.

    Args:
        Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(0)|O|φ_{b}(R)>.
        kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
        irvec (ndarray): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
        ndegen (ndarray, optional): number of degeneracy at each R.
        atoms_frac (ndarray, optional): atom's position in fractional coordinates.

    Returns:
        ndarray: k-space representation of the given operator, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
    """
    Or = np.array(Or, dtype=complex)
    irvec = np.array(irvec, dtype=float)
    Nr = irvec.shape[0]
    if ndegen is None:
        weight = np.array([1.0 for i in range(Nr)])
    else:
        ndegen = np.array(ndegen)
        weight = np.array([1.0 / ndegen[i] for i in range(Nr)])

    kpoints = np.array(kpoints, dtype=float)

    kR = np.einsum("ka,Ra->kR", kpoints, irvec, optimize=True)
    phase_R = np.exp(+2 * np.pi * 1j * kR)

    if atoms_frac is not None:
        tau = np.array(atoms_frac)
        ktau = np.einsum("ka,ma->km", kpoints, tau, optimize=True)
        eiktau = np.exp(-2 * np.pi * 1j * ktau)

        Ok = np.einsum("R,kR,km,Rmn,kn->kmn", weight, phase_R, eiktau, Or, eiktau.conjugate(), optimize=True)
    else:
        Ok = np.einsum("R,kR,Rmn->kmn", weight, phase_R, Or, optimize=True)

    return Ok


# ==================================================
def fourier_transform_r_to_k_vec(
    Or_vec, kpoints, irvec, ndegen=None, atoms_frac=None, unit_cell_cart=None, pseudo=False
):
    """
    fourier transformation of an arbitrary operator from real-space representation into k-space representation.

    Args:
        Or_vec (ndarray): real-space representation of the given operator, [O_{ab}^{x}(R), O_{ab}^{y}(R), O_{ab}^{z}(R)].
        kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
        irvec (ndarray): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
        ndegen (ndarray, optional): number of degeneracy at each R.
        atoms_frac (ndarray, optional): atom's position in fractional coordinates.
        unit_cell_cart (ndarray): transform matrix, [a1,a2,a3], [None].
        pseudo (bool, optional): calculate pseudo vector?

    Returns:
        ndarray: k-space representation of the given operator, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
    """
    Or_vec = np.array(Or_vec, dtype=complex)
    irvec = np.array(irvec, dtype=float)
    Nr = irvec.shape[0]
    if ndegen is None:
        weight = np.array([1.0 for _ in range(Nr)])
    else:
        ndegen = np.array(ndegen)
        weight = np.array([1.0 / ndegen[i] for i in range(Nr)])
    kpoints = np.array(kpoints, dtype=float)

    kR = np.einsum("ka,Ra->kR", kpoints, irvec, optimize=True)
    phase_R = np.exp(+2 * np.pi * 1j * kR)

    if atoms_frac is not None:
        tau = np.array(atoms_frac)
        ktau = np.einsum("ka,ma->km", kpoints, tau, optimize=True)
        eiktau = np.exp(-2 * np.pi * 1j * ktau)

        Ok_true_vec = np.einsum(
            "R,kR,km,aRmn,kn->akmn", weight, phase_R, eiktau, Or_vec, eiktau.conjugate(), optimize=True
        )
    else:
        Ok_true_vec = np.einsum("R,kR,aRmn->akmn", weight, phase_R, Or_vec, optimize=True)

    if not pseudo:
        return Ok_true_vec
    else:
        A = np.array(unit_cell_cart)
        irvec_cart = np.array([np.array(R) @ np.array(A) for R in irvec])

        Ok_pseudo_vec = np.zeros(Ok_true_vec.shape, dtype=np.complex128)
        ab_list = [(1, 2), (2, 0), (0, 1)]
        if atoms_frac is not None:
            atoms_cart = np.array([np.array(r) @ np.array(A) for r in atoms_frac])
            bond_cart = np.array([[[(R + rn) - rm for rn in atoms_cart] for rm in atoms_cart] for R in irvec_cart])
            for c, (a, b) in enumerate(ab_list):
                Ok_pseudo_vec[c] = 1.0j * np.einsum(
                    "R,kR,Rmn,km,Rmn,kn->kmn",
                    weight,
                    phase_R,
                    bond_cart[:, :, :, a],
                    eiktau,
                    Or_vec[b],
                    eiktau.conjugate(),
                    optimize=True,
                )
                -1.0j * np.einsum(
                    "R,kR,Rmn,km,Rmn,kn->kmn",
                    weight,
                    phase_R,
                    bond_cart[:, :, :, b],
                    eiktau,
                    Or_vec[a],
                    eiktau.conjugate(),
                    optimize=True,
                )
        else:
            for c, (a, b) in enumerate(ab_list):
                Ok_pseudo_vec[c] = 1.0j * np.einsum(
                    "R,kR,R,Rmn->kmn", weight, phase_R, irvec_cart[:, a], Or_vec[b], optimize=True
                ) - 1.0j * np.einsum("R,kR,R,Rmn->kmn", weight, phase_R, irvec_cart[:, b], Or_vec[a], optimize=True)

        return Ok_true_vec, Ok_pseudo_vec


# ==================================================
def fourier_transform_r_to_k_new(Or, kpoints, unit_cell_cart, irvec, ndegen=None, atoms_frac=None):
    """
    fourier transformation of an arbitrary operator from real-space representation into k-space representation.

    Args:
        Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(0)|O|φ_{b}(R)>.
        kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
        unit_cell_cart (ndarray): transform matrix, [a1,a2,a3], [None].
        irvec (ndarray): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
        ndegen (ndarray, optional): number of degeneracy at each R.
        atoms_frac (ndarray, optional): atom's position in fractional coordinates.

    Returns:
        ndarray: k-space representation of the given operator, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
    """
    Or = np.array(Or, dtype=complex)
    irvec = np.array(irvec, dtype=float)
    Nr = irvec.shape[0]
    if ndegen is None:
        weight = np.array([1.0 for _ in range(Nr)])
    else:
        ndegen = np.array(ndegen)
        weight = np.array([1.0 / ndegen[i] for i in range(Nr)])
    kpoints = np.array(kpoints, dtype=float)

    kR = np.einsum("ka,Ra->kR", kpoints, irvec, optimize=True)
    phase_R = np.exp(+2 * np.pi * 1j * kR)

    A = np.array(unit_cell_cart)
    irvec_cart = np.array([np.array(R) @ np.array(A) for R in irvec])

    if atoms_frac is not None:
        tau = np.array(atoms_frac)
        ktau = np.einsum("ka,ma->km", kpoints, tau, optimize=True)
        eiktau = np.exp(-2 * np.pi * 1j * ktau)

        atoms_cart = np.array([np.array(r) @ np.array(A) for r in atoms_frac])

        bond_cart = np.array([[[(R + rn) - rm for rn in atoms_cart] for rm in atoms_cart] for R in irvec_cart])

        Ok = np.einsum("R,kR,km,Rmn,kn->kmn", weight, phase_R, eiktau, Or, eiktau.conjugate(), optimize=True)
        Ok_dx, Ok_dy, Ok_dz = 1.0j * np.einsum(
            "R,kR,Rmna,km,Rmn,kn->akmn", weight, phase_R, bond_cart, eiktau, Or, eiktau.conjugate(), optimize=True
        )
    else:
        Ok = np.einsum("R,kR,Rmn->kmn", weight, phase_R, Or, optimize=True)
        Ok_dx, Ok_dy, Ok_dz = 1.0j * np.einsum("R,kR,Ra,Rmn->akmn", weight, phase_R, irvec_cart, Or, optimize=True)

    delOk = np.array([Ok_dx, Ok_dy, Ok_dz])

    return Ok, delOk


# ==================================================
def interpolate(Ok, kpoints_0, kpoints, irvec, ndegen=None, atoms_frac=None):
    """
    interpolate an arbitrary operator by implementing
    fourier transformation from real-space representation into k-space representation.

    Args:
        Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
        kpoints_0 (ndarray): k points before interpolated (crystal coordinate, [[k1,k2,k3]]).
        kpoints (ndarray): k points after interpolated (crystal coordinate, [[k1,k2,k3]]).
        irvec (ndarray): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
        ndegen (ndarray, optional): number of degeneracy at each R.
        atoms_frac (ndarray, optional): atom's position in fractional coordinates.

    Returns:
        ndarray: matrix elements at each k point, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
    """
    Or = fourier_transform_k_to_r(Ok, kpoints_0, irvec, atoms_frac)
    Ok_interpolated = fourier_transform_r_to_k(Or, kpoints, irvec, ndegen, atoms_frac)

    return Ok_interpolated


# ==================================================
def matrix_dict_r(Or, rpoints, diagonal=False):
    """
    dictionary form of an arbitrary operator matrix in real-space representation.

    Args:
        Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(0)|O|φ_{b}(R)>.
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
def sort_ket_matrix(Ok, ket1, ket2):
    """
    sort ket1 to align with the order of ket2.

    Args:
        Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|H|φ_{b}(k)>.
        ket1 (list): ket basis list, orbital@site.
        ket2 (list): ket basis list, orbital@site.

    Returns:
        Ok (ndarray): operator.
    """
    idx_list = [ket1.index(o) for o in ket2]
    Ok = Ok[:, idx_list, :]
    Ok = Ok[:, :, idx_list]

    return Ok


# ==================================================
def sort_ket_matrix_dict(Or_dict, ket, ket_samb):
    """
    sort ket to align with the SAMB definition (ket_samb).

    Args:
        Or_dict (dict): dictionary form of an arbitrary operator matrix in reak-space/k-space representation.
        ket (list): ket basis list, orbital@site.
        ket_samb (list): ket basis list for SAMBs, orbital@site.

    Returns:
        Or_dict (dict):  dictionary form of an arbitrary operator matrix.
    """
    idx_list = [ket_samb.index(o) for o in ket]
    Or_dict = {(n1, n2, n3, idx_list[a], idx_list[b]): v for (n1, n2, n3, a, b), v in Or_dict.items()}

    return Or_dict


# ==================================================
def sort_ket_list(lst, ket, ket_samb):
    """
    sort ket to align with the SAMB definition (ket_samb).

    Args:
        lst (list): arbitrary list that has the same dimensions as the ket.
        ket (list): ket basis list, orbital@site.
        ket_samb (list): ket basis list for SAMBs, orbital@site.

    Returns:
        lst (list): arbitrary list that has the same dimensions as the ket.
    """
    idx_list = [ket.index(o) for o in ket_samb]

    lst = np.array(lst)

    if lst.ndim == 1:
        lst = list(np.array(lst)[idx_list])
    elif lst.ndim == 2:
        lst = list(np.array(lst)[idx_list, :])
    elif lst.ndim == 3:
        lst = list(np.array(lst)[idx_list, :, :])
    else:
        raise Exception(f"invalid dimension of lst = {lst.ndim} was given.")

    return list(lst)


# ==================================================
def samb_decomp_operator(
    Or_dict, Zr_dict, A=None, atoms_frac=None, ket=None, A_samb=None, atoms_frac_samb=None, ket_samb=None
):
    """
    decompose arbitrary operator into linear combination of SAMBs.

    Args:
        Or_dict (dict): dictionary form of an arbitrary operator matrix in real-space/k-space representation.
        Zr_dict (dict): dictionary form of SAMBs.
        A (list/ndarray, optional): real lattice vectors for the given operator, A = [a1,a2,a3] (list), [[[1,0,0], [0,1,0], [0,0,1]]].
        atoms_frac (ndarray, optional): atom's position in fractional coordinates for the given operator.
        ket (list, optional): ket basis list, orbital@site.
        A_samb (list/ndarray, optional): real lattice vectors for SAMBs, A = [a1,a2,a3] (list), [[[1,0,0], [0,1,0], [0,0,1]]].
        atoms_frac_samb (ndarray, optional): atom's position in fractional coordinates for SAMBs.
        ket_samb (list, optional): ket basis list for SAMBs, orbital@site.

    Returns:
        z (dict): parameter set, {tag: z_j}.
    """
    Or_dict = sort_ket_matrix_dict(Or_dict, ket, ket_samb)

    # if A is not None:
    #     if not np.allclose(A, A_samb, rtol=1e-03, atol=1e-03):
    # if True:
    #     if True:
    #         print("ok!!!")
    #         A = np.array(A, dtype=float)
    #         A_samb = np.array(A_samb, dtype=float)
    #         atoms_frac = np.array(sort_ket_list(atoms_frac, ket, ket_samb), dtype=float)
    #         atoms_frac_samb = np.array(atoms_frac_samb, dtype=float)

    #         Or_dict_ = {}
    #         for (R1, R2, R3, m, n), v in Or_dict.items():
    #             if np.abs(v) < 0.0:
    #                 continue

    #             R = np.array([R1, R2, R3], dtype=float)
    #             rm = atoms_frac[m]
    #             rn = atoms_frac[n]
    #             bond = ((R + rn) - rm) @ A

    #             bond = tuple([format(bi, ".3f") for bi in bond])
    #             bond = tuple([float(bi) for bi in bond])

    #             Or_dict_[(*bond, m, n)] = v

    #         Or_dict = Or_dict_

    #         Zr_dict_ = {}
    #         for tag, d in Zr_dict.items():
    #             dic = {}
    #             for (R1, R2, R3, m, n), v in d.items():
    #                 if np.abs(v) < 0.0:
    #                     continue

    #                 R = np.array([R1, R2, R3], dtype=float)
    #                 rm = atoms_frac_samb[m]
    #                 rn = atoms_frac_samb[n]
    #                 bond = ((R + rn) - rm) @ A_samb

    #                 bond = tuple([format(bi, ".3f") for bi in bond])
    #                 bond = tuple([float(bi) for bi in bond])

    #                 dic[(*bond, m, n)] = v

    #             Zr_dict_[tag] = dic

    #         Zr_dict = Zr_dict_

    z = {
        tag: np.real(np.sum([v * Or_dict.get((-k[0], -k[1], -k[2], k[4], k[3]), 0) for k, v in d.items()]))
        for tag, d in Zr_dict.items()
    }

    return z


# ==================================================
def O_R_dependence(Or, A, irvec, ndegen, ef=0.0):
    """
    Bond length ||R|| (the 2-norm of lattice vector) dependence of the Frobenius norm of the operator ||O(R)||.
    The decay length τ [Ang] defined by Exponential-form fitting ||O(R)|| = ||O(Rmin)|| exp(-||R||/τ) is also returned.
    Rmin is the bond with minimum length.

    Args:
        Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(0)|O|φ_{b}(R)>.
        A (list/ndarray): real lattice vectors, A = [a1,a2,a3] (list), [[[1,0,0], [0,1,0], [0,0,1]]].
        irvec (ndarray, optional): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
        ndegen (ndarray, optional): number of degeneracy at each R.
        ef (float, optional): fermi energy.

    Returns: (list, list, list, float, float, float)
            [||R||], [||O(R)||], [max(|O(R)|)], ||O(Rmin)||, ||Rmin||, τ.
    """
    R_2_norm_lst = []
    OR_F_norm_lst = []
    OR_abs_max_lst = []

    a = np.linalg.norm(A[0])

    num_R = Or.shape[0]
    for iR in range(num_R):
        n1, n2, n3 = irvec[iR]
        R = np.array([n1, n2, n3]) @ A

        OR = Or[iR]
        OR = OR / ndegen[iR]

        if (n1, n2, n3) == (0, 0, 0):
            OR = OR - np.diag(OR.diagonal())

            if np.linalg.norm(OR, ord="fro") < 1e-8:
                continue

        R_2_norm = np.linalg.norm(R)
        OR_F_norm = np.linalg.norm(OR, ord="fro")
        OR_abs_max = np.max(np.abs(OR))

        R_2_norm_lst.append(R_2_norm)
        OR_F_norm_lst.append(OR_F_norm)
        OR_abs_max_lst.append(OR_abs_max)

    zip_lists = zip(R_2_norm_lst, OR_F_norm_lst, OR_abs_max_lst)
    zip_sort = sorted(zip_lists)
    R_2_norm_lst, OR_F_norm_lst, OR_abs_max_lst = zip(*zip_sort)

    R_2_norm_min = np.min(R_2_norm_lst)
    R_2_min_lst = list(np.where(R_2_norm_lst == R_2_norm_min))[0]
    Omin_F_norm = np.max(np.array(OR_F_norm_lst)[R_2_min_lst])

    coefficients = np.polyfit(R_2_norm_lst, -np.log((OR_F_norm_lst) / Omin_F_norm), 1)
    tau = 1.0 / coefficients[0]

    return R_2_norm_lst, OR_F_norm_lst, OR_abs_max_lst, Omin_F_norm, R_2_norm_min, tau


# ==================================================
def construct_Or(z, num_wann, rpoints, matrix_dict):
    """
    arbitrary operator constructed by linear combination of SAMBs in real-space representation.

    Args:
        z (list): parameter set, [z_j].
        num_wann (int): # of WFs.
        rpoints (ndarray, optional): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
        matrix_dict (dict): SAMBs.

    Returns:
        ndarray: matrix, [#r, dim, dim].
    """
    Or_dict = {(n1, n2, n3, a, b): 0.0 for (n1, n2, n3) in rpoints for a in range(num_wann) for b in range(num_wann)}
    for j, d in enumerate(matrix_dict["matrix"].values()):
        zj = z[j]
        for (n1, n2, n3, a, b), v in d.items():
            if (n1, n2, n3, a, b) in Or_dict:
                Or_dict[(n1, n2, n3, a, b)] += zj * v

    Or = np.array(
        [
            [[Or_dict.get((n1, n2, n3, a, b), 0.0) for b in range(num_wann)] for a in range(num_wann)]
            for (n1, n2, n3) in rpoints
        ]
    )

    return Or


# ==================================================
def construct_Ok(z, num_wann, kpoints, rpoints, matrix_dict, atoms_frac=None):
    """
    arbitrary operator constructed by linear combination of SAMBs in k-space representation.

    Args:
        z (list): parameter set, [z_j].
        num_wann (int): # of WFs.
        kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
        rpoints (ndarray, optional): lattice points (crystal coordinate, [[n1,n2,n3]], nj: integer).
        matrix_dict (dict): SAMBs.

    Returns:
        ndarray: matrix, [#k, dim, dim].
    """
    kpoints = np.array(kpoints)
    rpoints = np.array(rpoints)

    Or = construct_Or(z, num_wann, rpoints, matrix_dict)
    Ok = fourier_transform_r_to_k(Or, kpoints, rpoints, atoms_frac=atoms_frac)

    return Ok


# ==================================================
def thermal_avg(O, E, U, ef=0.0, T_Kelvin=0.0, num_k=0):
    """
    thermal average of the given operator,
    <O> = 1 / Nk * sum_{n,k} fermi_dirac[E_{n}(k)] O_{nn}(k)

    Args:
        O (ndarray): operator or list of operator.
        E (ndarray): eigen values.
        U (ndarray): eigen vectors.
        ef (float, optional): fermi energy.
        T (float, optional): temperature.

    Returns:
        ndarray: thermal average of the given operator.
    """
    if type(O) != list:
        single_operator = True
        O = [O]
    else:
        single_operator = False

    if ef is None:
        ef = 0.0

    fk = fermi(E - ef, T_Kelvin)

    O = np.array(O)
    E = np.array(E)
    U = np.array(U)

    if num_k == 0:
        num_k = E.shape[0]

    O_exp = []
    for i, Oi in enumerate(O):
        UdoU = U.transpose(0, 2, 1).conjugate() @ Oi @ U
        UdoU_diag = np.diagonal(UdoU, axis1=1, axis2=2)
        Oi_exp = np.sum(fk * UdoU_diag) / num_k
        O_exp.append(np.real(Oi_exp))

        if np.imag(Oi_exp) > 1e-7:
            raise Exception(f"expectation value of {i+1}th operator is wrong : {Oi_exp}")

    if single_operator:
        O_exp = O_exp[0]
    else:
        O_exp = np.array(O_exp)

    return O_exp


# ==================================================
def total_energy(E, ef=0.0, T_Kelvin=0.0, num_k=0):
    """
    total energy (E).
    """
    if ef is None:
        ef = 0.0

    E = np.array(E)
    fk = fermi(E - ef, T_Kelvin)

    if num_k == 0:
        num_k = E.shape[0]

    E_tot = np.sum(fk * E) / num_k

    return E_tot


# ==================================================
def free_energy(E, ef=0.0, T_Kelvin=0.0, num_k=0):
    """
    free energy (F = E - TS)
    """
    if ef is None:
        ef = 0.0

    E = np.array(E)
    fk = fermi(E - ef, T_Kelvin)

    if num_k == 0:
        num_k = E.shape[0]

    T_eV = Kelvin_to_eV(T_Kelvin)
    F = -T_eV * np.sum((-1.0 / T_eV * (E - ef) - np.log(fk))) / num_k

    return F


# ==================================================
def entropy(E, ef=0.0, T_Kelvin=0.0, num_k=0):
    """
    total energy (S).
    """
    if ef is None:
        ef = 0.0

    if num_k == 0:
        num_k = E.shape[0]

    fk = fermi(E - ef, T_Kelvin)

    log_fk = np.array([[np.log(fkin) if fkin != 0.0 else 0.0 for fkin in fki] for fki in fk])
    log_1mfk = np.array([[np.log(1.0 - fkin) if 1.0 - fkin != 0.0 else 0.0 for fkin in fki] for fki in fk])

    S = -np.sum(fk * log_fk + (1.0 - fk) * log_1mfk) / num_k

    return S


# ==================================================
def Rx(theta):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def Ry(theta):
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def Rz(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


# ==================================================
def pauli_spn_x(dim):
    Identity = np.eye(int(dim / 2))
    sig_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    return TensorProduct(Identity, sig_x)


# ==================================================
def pauli_spn_y(dim):
    Identity = np.eye(int(dim / 2))
    sig_y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    return TensorProduct(Identity, sig_y)


# ==================================================
def pauli_spn_z(dim):
    Identity = np.eye(int(dim / 2))
    sig_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    return TensorProduct(Identity, sig_z)


# ==================================================
def spn_operator(pauli_spn=None, g_factor=2.0, dim=2):
    if pauli_spn is None:
        pauli_spn = np.array([pauli_spn_x(dim), pauli_spn_y(dim), pauli_spn_z(dim)])

    mu_B = bohr_magn_SI * joul_to_eV
    return -0.5 * g_factor * mu_B * pauli_spn


# ==================================================
def spin_zeeman_interaction(B, theta=0.0, phi=0.0, pauli_spn=None, g_factor=2.0, dim=2):
    spin_oper = spn_operator(pauli_spn, g_factor, dim)
    B_vec = B * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    H_zeeman = -sum([spin_oper[i] * B_vec[i] for i in range(3)])

    return H_zeeman


# ==================================================
def write_or(outdir, filename, unit_cell_cart, Or, irvec, ndegen=None, vec=False):
    """
    write seedname_or.dat.

    Args:
        outdir (str): input and output files are found in this directory.
        filename (str): file name.
        unit_cell_cart (ndarray): transform matrix, [a1,a2,a3], [None].
        Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(0)|O|φ_{b}(R)>.
        irvec (ndarray): irreducible rpoints.
        ndegen (ndarray): number of degeneracy at each R.
        vec (bool, optional): vector ?
    """
    Or = np.array(Or)
    num_wann = Or.shape[-1]

    fs = open(f"{outdir}/{filename}", "w")
    fs.write("# written {}\n".format(datetime.datetime.now().strftime("on %d%b%Y at %H:%M:%S")))

    Or_str = " {0[0]:18.15f} {0[1]:18.15f} {0[2]:18.15f}\n".format(unit_cell_cart[0, :])
    Or_str += " {0[0]:18.15f} {0[1]:18.15f} {0[2]:18.15f}\n".format(unit_cell_cart[1, :])
    Or_str += " {0[0]:18.15f} {0[1]:18.15f} {0[2]:18.15f}\n".format(unit_cell_cart[2, :])

    if ndegen is not None:
        Or_str += "{:12d}\n{:12d}\n".format(num_wann, len(ndegen))
        Or_str += textwrap.fill("".join(["{:5d}".format(x) for x in ndegen]), 75, drop_whitespace=False)
        Or_str += "\n"
    else:
        Or_str += "{:12d}\n".format(num_wann)

    for irpts in range(len(irvec)):
        for i, j in itertools.product(range(num_wann), repeat=2):
            v = irvec[irpts, :]
            line = "{:5d}{:5d}{:5d}{:5d}{:5d}  ".format(
                int(round(v[0])), int(round(v[1])), int(round(v[2])), j + 1, i + 1
            )
            if vec:
                line += "".join([" {:>15.8E}  {:>15.8E}".format(x.real, x.imag) for x in Or[:, irpts, j, i]])
            else:
                x = Or[irpts, j, i]
                line += " {:>15.8E}  {:>15.8E}".format(x.real, x.imag)
            line += "\n"

            Or_str += line

    fs.write(Or_str)
