"""
tool codes for seedname_hr.dat.
"""

import numpy as np


# ==================================================
def read_hr(filename, orb_dict=None, encoding="UTF-8"):
    """
    read seedname_hr.dat file which is obtained by Wannier90.
    seedname_hr.dat file includes the wannier tight-binding Hamiltoinan's matrix elements in real space, H_{mn}(R).

    Args:
        filename (str): file name of seedname_hr.dat file.
        orb_dict (dict): dictionary of orbitals information, { wannier orbital #: pseudo atomic orbital # }.
        encoding (str, optional): encoding.

    Returns:
        tuple: (HH_R, irvec, ndegen)
        - HH_R (ndarray): H_{mn}(R).
        - irvec (ndarray): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
        - ndegen (ndarray): number of degeneracy at each R.

    Notes:
        - R = n1 a1 + n2 a2 + n3 a3 (aj: lattice vectors, nj: integer)
        - m: wannier orbital # starting from 1.
        - n: pseudo atomic orbital # starting from 1.
    """
    line_block = 15

    f = open(filename, "r", encoding=encoding)
    data = f.readlines()
    f.close()

    dim = int(data[1].rstrip("\n").strip())
    num_R = int(data[2].rstrip("\n").strip())
    m = num_R % line_block

    data = data[3:]
    data = [lst.rstrip("\n").split(" ") for lst in data]
    data = [[v for v in lst if v != ""] for lst in data]

    ndegen = [lst for lst in data if len(lst) == line_block]
    ndegen = [int(v) for lst in ndegen for v in lst]

    hr_data = [lst for lst in data if len(lst) != line_block]
    if m != 0:
        ndegen += [int(v) for lst in hr_data[0] for v in lst]
        hr_data = hr_data[1:]

    hr_data = [[float(v) if "." in v else int(v) for v in lst] for lst in hr_data if lst != []]
    hr_dict = {(n1, n2, n3, m, n): (real, imag) for n1, n2, n3, m, n, real, imag in hr_data}

    irvec = [(n1, n2, n3) for n1, n2, n3, _, _ in hr_dict.keys()]
    irvec = sorted(set(irvec), key=irvec.index)

    HH_R_dict = {(n1, n2, n3): np.zeros((dim, dim), dtype=complex) for n1, n2, n3, _, _ in hr_dict.keys()}

    for k, v in hr_dict.items():
        n1, n2, n3, m, n = k
        re, im = v
        m = m - 1
        if orb_dict is not None:
            n = orb_dict[n] - 1
        else:
            n = n - 1
        HH_R_dict[(n1, n2, n3)][m][n] += re + 1j * im

    for v in HH_R_dict.keys():
        if (-v[0], -v[1], -v[2]) not in HH_R_dict:
            raise Exception("invalid")

    HH_R = np.array([HH_R_dict[(n1, n2, n3)] for n1, n2, n3 in irvec], dtype=complex)

    return HH_R, irvec, ndegen


# ==================================================
def remove_elements(Hr_mat, idx_list):
    """
    read seedname_hr.dat file which is obtained by Wannier90.
    seedname_hr.dat file includes the wannier tight-binding Hamiltoinan's matrix elements in real space, H_{mn}(R).

    Args:
        Hr_mat (dict): H_{mn}(R) { (n1,n2,n3,m,n): (re,im) }.
        idx_list (list): index list to be removed.

    Returns:
        - dict: H_{mn}(R) { (n1,n2,n3,m,n): (re,im) }

    Notes:
        - R = n1 a1 + n2 a2 + n3 a3 (aj: lattice vectors, nj: integer)
        - m: wannier orbital # starting from 1.
        - n: pseudo atomic orbital # starting from 1.
    """
    Hr_mat_removed = {}
    for (n1, n2, n3), mat in Hr_mat.items():
        Hr_mat_removed[(n1, n2, n3)] = mat
        for idx in idx_list:
            Hr_mat_removed[(n1, n2, n3)][idx, :] = 0.0
            Hr_mat_removed[(n1, n2, n3)][:, idx] = 0.0

    return Hr_mat_removed


# ==================================================
def convert_hr_to_hk(k, Hr_mat, R_degen_dict, dim):
    """
    create hamiltonian matrix from wanner90 output file.

    Args:
        k (np.array): k grid.
        Hr_mat (dict): H_{mn}(R) { (n1,n2,n3,m,n): (re,im) }.
        R_degen_dict (dict): degeneracy(R) { (n1,n2,n3): deg }.
        dim (int): matrix dimension.

    Returns:
        np.array: a set of hamiltonian matrices
    """
    hk = np.zeros((dim, dim, k.shape[0]), dtype=complex)
    for idx, m in Hr_mat.items():
        n1, n2, n3 = idx
        deg = R_degen_dict[(n1, n2, n3)]
        nn = np.array([n1, n2, n3])
        p = 2 * np.pi * k @ nn
        e = np.exp(1j * p)
        m = np.broadcast_to(m, (k.shape[0], dim, dim)).transpose((1, 2, 0))
        hk += (m / deg) * e

    hk = hk.transpose((2, 0, 1))

    return hk
