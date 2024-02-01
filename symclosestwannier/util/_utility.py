"""
utility codes.
"""
import numpy as np
import fortio, scipy.io

from gcoreutils.nsarray import NSArray

M_ZERO = np.finfo(float).eps


# ==================================================
class FortranFileR(fortio.FortranFile):
    def __init__(self, filename):
        try:
            super().__init__(filename, mode="r", header_dtype="uint32", auto_endian=True, check_file=True)
        except ValueError:
            print("File '{}' contains subrecords - using header_dtype='int32'".format(filename))
            super().__init__(filename, mode="r", header_dtype="int32", auto_endian=True, check_file=True)


# ==================================================
class FortranFileW(scipy.io.FortranFile):
    def __init__(self, filename):
        super().__init__(filename, mode="w")


# ==================================================
def is_zero(x):
    return np.abs(x) < M_ZERO * 100


# ==================================================
def fermi(x, T=0.01):
    if T == 0.0:
        return x < 0.0

    return 0.5 * (1.0 - np.tanh(0.5 * x / T))


# ==================================================
def fermi_dt(x, T=0.01):
    return fermi(x, T) * fermi(-x, T) / T


# ==================================================
def fermi_ddt(x, T=0.01):
    return (1 - 2 * fermi(-x, T)) * fermi(x, T) * fermi(-x, T) / T / T


# ==================================================
def weight_proj(e, e0, e1, T0, T1, delta=10e-12):
    """weight function for projection"""
    return fermi(e0 - e, T0) + fermi(e - e1, T1) - 1.0 + delta


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
def wigner_seitz(A, mp_grid, prec=1.0e-7):
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

    irvec = np.array(irvec)
    ndegen = np.array(ndegen)

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
        (ndarray, ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>, lattice points.
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
        Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>.
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
def fourier_transform_r_to_k_vec(Or_vec, kpoints, irvec, ndegen=None, atoms_frac=None):
    """
    fourier transformation of an arbitrary operator from real-space representation into k-space representation.

    Args:
        Or_vec (ndarray): real-space representation of the given operator, [O_{ab}^{x}(R), O_{ab}^{y}(R), O_{ab}^{z}(R)].
        kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
        irvec (ndarray): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
        ndegen (ndarray, optional): number of degeneracy at each R.
        atoms_frac (ndarray, optional): atom's position in fractional coordinates.

    Returns:
        ndarray: k-space representation of the given operator, O_{ab}(k) = <φ_{a}(k)|O|φ_{b}(k)>.
    """
    Ok_x = fourier_transform_r_to_k(Or_vec[0], kpoints, irvec, ndegen, atoms_frac)
    Ok_y = fourier_transform_r_to_k(Or_vec[1], kpoints, irvec, ndegen, atoms_frac)
    Ok_z = fourier_transform_r_to_k(Or_vec[2], kpoints, irvec, ndegen, atoms_frac)

    Ok_vec = np.array([Ok_x, Ok_y, Ok_z])

    return Ok_vec


# ==================================================
def fourier_transform_r_to_k_new(Or, kpoints, unit_cell_cart, irvec, ndegen=None, atoms_frac=None):
    """
    fourier transformation of an arbitrary operator from real-space representation into k-space representation.

    Args:
        Or (ndarray): real-space representation of the given operator, O_{ab}(R) = <φ_{a}(R)|O|φ_{b}(0)>.
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
        print(atoms_cart.shape)

        bond_cart = np.array([[[((R + rm) - rn) for rn in atoms_cart] for rm in atoms_cart] for R in irvec_cart])

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
def sort_ket_matrix_k(Ok, ket, ket_samb):
    """
    sort ket to align with the SAMB definition (ket_samb).

    Args:
        Ok (ndarray): arbitrary operator in k-space representation, O_{ab}(k) = <φ_{a}(k)|H|φ_{b}(k)>.
        ket (list): ket basis list, orbital@site.
        ket_samb (list): ket basis list for SAMBs, orbital@site.

    Returns:
        Ok (ndarray): operator.
    """
    idx_list = [ket.index(o) for o in ket_samb]
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
        Or_dict (dict): dictionary form of an arbitrary operator matrix in reak-space/k-space representation.
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

    if A is None or (A is not None and np.allclose(A, A_samb, rtol=1e-05, atol=1e-05)):
        z = {
            tag: np.real(np.sum([v * Or_dict.get((-k[0], -k[1], -k[2], k[4], k[3]), 0) for k, v in d.items()]))
            for tag, d in Zr_dict.items()
        }
    else:
        A = np.array(A, dtype=float)
        A_samb = np.array(A_samb, dtype=float)
        atoms_frac = np.array(sort_ket_list(atoms_frac, ket, ket_samb), dtype=float)
        atoms_frac_samb = np.array(atoms_frac_samb, dtype=float)

        Or = {}
        for (R1, R2, R3, m, n), v in Or_dict.items():
            if np.abs(v) < 1e-12:
                continue

            R = np.array([R1, R2, R3], dtype=float)
            rm = atoms_frac[m]
            rn = atoms_frac[n]
            bond = (R + rm - rn) @ A

            if (m, n) in Or:
                Or[(m, n)].append((bond, R1, R2, R3, v))
            else:
                Or[(m, n)] = [(bond, R1, R2, R3, v)]

        Zr_dict_ = {}
        for tag, d in Zr_dict.items():
            dic = {}
            for (R1, R2, R3, m, n), v in d.items():
                if np.abs(v) < 1e-12:
                    continue

                R = np.array([R1, R2, R3], dtype=float)
                rm = atoms_frac_samb[m]
                rn = atoms_frac_samb[n]
                bond = (R + rm - rn) @ A_samb

                if (m, n) in dic:
                    dic[(m, n)].append((bond, R1, R2, R3, v))
                else:
                    dic[(m, n)] = [(bond, R1, R2, R3, v)]

            Zr_dict_[tag] = dic

        z = {
            tag: np.sum(
                [
                    np.real(v * v_)
                    for m, n in d.keys()
                    for bond, R1, R2, R3, v in d[(m, n)]
                    for bond_, R1_, R2_, R3_, v_ in Or[(n, m)]
                    if np.allclose(bond, -bond_, rtol=1e-03, atol=1e-03)
                ]
            )
            for tag, d in Zr_dict_.items()
        }

    return z


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
    cell_site = matrix_dict["cell_site"]
    ket = matrix_dict["ket"]
    atoms_frac = [
        NSArray(cell_site[ket[a].split("@")[1]][0], style="vector", fmt="value").tolist() for a in range(len(ket))
    ]

    Or = construct_Or(z, num_wann, rpoints, matrix_dict)
    Ok = fourier_transform_r_to_k(Or, kpoints, rpoints, atoms_frac=atoms_frac)

    return Ok


# ==================================================
def thermal_avg(O, E, U, ef=0.0, T=0.0):
    """
    thermal average of the given operator,
    <O> = 1 / Nk * \sum_{n,k} fermi_dirac[E_{n}(k)] O_{nn}(k)

    Args:
        O (ndarray): operator or list of operator.
        E (ndarray): eigen values.
        U (ndarray): eigen vectors.
        ef (float, optional): fermi energy.
        T (float, optional): temperature.

    Returns:
        ndarray: thermal average of the given operator.
    """
    fk = fermi(E - ef, T)

    O = np.array(O)
    E = np.array(E)
    U = np.array(U)

    num_k = E.shape[0]

    if O.ndim == 2:
        O = np.array([O])

    O_exp = []
    for i, Oi in enumerate(O):
        UdoU = U.transpose(0, 2, 1).conjugate() @ Oi @ U
        UdoU_diag = np.diagonal(UdoU.transpose(1, 2, 0))
        Oi_exp = np.sum(fk * UdoU_diag) / num_k
        O_exp.append(np.real(Oi_exp))

        if np.imag(Oi_exp) > 1e-7:
            raise Exception(f"expectation value of {i+1}th operator is wrong : {Oi_exp}")

    O_exp = np.array(O_exp)

    return O_exp
