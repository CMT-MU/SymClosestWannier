"""
utility codes for reading input and DFT output.
"""
import os
import xmltodict
import numpy as np


_default_proj_min = 0.0
_default_smearing_temp_max = 5.0
_default_smearing_temp_min = 0.01
_default_delta = 1e-12
_default_a = None
_default_N1 = 50


# ==================================================
def cwin_reader(topdir, seedname="cwannier", encoding="UTF-8"):
    """
    read seedname.cwin file.

    Args:
        topdir (str): directory of seedname.cwin file.
        seedname (str): seedname.
        encoding (str, optional): encoding.

    Returns:
        dict: dictionary form of seedname.cwin.
            - restart*           : the restart position (str), ["wannierize"].
            - outdir*            : input and output files are found in this directory (str), ["./"].
            - disentangle*       : disentagle bands ? (bool), [False].
            - proj_min*          : minimum value of projectability: [0.0].
            - dis_win_emax*      : upper energy window (float), [None].
            - dis_win_emin*      : lower energy window (float), [None].
            - smearing_temp_max* : smearing temperature for upper window (float), [5.0].
            - smearing_temp_min* : smearing temperature for lower window (float), [0.01].
            - delta*             : small constant to avoid ill-conditioning of overlap matrices (float), [1e-12].
            - svd*               : implement singular value decomposition ? otherwise adopt Lowdin's orthogonalization method (bool), [False].
            - verbose*           : verbose calculation info (bool, optional), [False].
            - write_hr*          : write seedname_hr.py ? (bool), [False].
            - write_sr*          : write seedname_sr.py ? (bool), [False].

        # only used for symmetrization.
            - symmetrization*    : symmetrize ? (bool), [False].
            - mp_outdir*         : output files for multipie are found in this directory (str). ["./"].
            - mp_seedname*       : seedname for seedname_model.py, seedname_samb.py and seedname_matrix.py files (str).
            - ket_amn*           : ket basis list in the seedname.amn file. The format of each ket must be same as the "ket" in sambname_model.py file. See sambname["info"]["ket"] in sambname_model.py file for the format (list), [None].
            - irreps*            : list of irreps to be considered (str/list), [None].

        # only used for band dispersion calculation.
            - a*                 : lattice parameter (in Ang) used to correct units of k points in reference band data, [1.0].
            - fermi_energy*      : fermi energy used for band shift, [None].
            - N1*                : number of divisions for high symmetry lines (int, optional), [50].
    """
    # default
    model_dict = {
        "restart": "wannierise",
        "outdir": "./",
        "seedname": seedname,
        #
        "disentangle": False,
        "proj_min": _default_proj_min,
        "dis_win_emax": None,
        "dis_win_emin": None,
        "smearing_temp_max": _default_smearing_temp_max,
        "smearing_temp_min": _default_smearing_temp_min,
        "delta": _default_delta,
        "svd": False,
        #
        "verbose": False,
        #
        "write_hr": False,
        "write_sr": False,
        #
        "symmetrization": False,
        "mp_outdir": "./",
        "mp_seedname": seedname,
        "ket_amn": None,
        "irreps": "all",
        #
        "a": _default_a,
        "N1": _default_N1,
        "fermi_energy": None,
    }

    def _str_to(k, v):
        v = str(v).replace("'", "").replace('"', "")

        if k in ("seedname", "mp_seedname"):
            pass
        elif k in ("outdir", "mp_outdir"):
            v = v[:-1] if v[-1] == "/" else v
        elif k == "restart":
            if v not in ("wannierise", "symmetrization"):
                raise Exception(f"invalid restart = {v} was given. choose from 'wannierise'/'symmetrization'.")
        elif k in (
            "proj_min",
            "dis_win_emax",
            "dis_win_emin",
            "smearing_temp_max",
            "smearing_temp_min",
            "delta",
            "a",
            "fermi_energy",
        ):
            v = float(v)
        elif k == "N1":
            v = int(v)
        elif k == "ket_amn":
            if "(" in str(v) and ")" in str(v):
                v = [str(o) if i == 0 else f"({str(o)}" for i, o in enumerate(v[1:-1].split(",("))]
            else:
                v = [str(o) for o in v[1:-1].split(",")]
        elif k == "irreps":
            if "[" in v and "]" in v:
                v = [str(o) for o in v[1:-1].split(",")]
            else:
                if v not in ("all", "full"):
                    raise Exception(f"invalid irreps = {v} was given. choose from 'all'/'full'.")
        else:
            if v.lower() in ("true", ".true."):
                v = True
            elif v.lower() in ("false", ".false."):
                v = False
            else:
                raise Exception(f"invalid {k} = {v} was given. choose from 'true'/'.true.'/'false'/'.false.'.")

        return v

    topdir = topdir[:-1] if topdir[-1] == "/" else topdir
    f = open(topdir + "/" + seedname + ".cwin", "r", encoding=encoding)
    data = f.readlines()
    f.close()

    for line in data:
        line = [vi for vi in line.replace("\n", "").split(" ") if vi != ""]
        if len(line) == 0:
            continue

        if "!" in line[0]:
            continue

        if "#" in line[0]:
            continue

        k = line[0]
        if k == "ket_amn":
            v = "".join(line[2:]) if line[1] == "=" else "".join(line[1:])
        else:
            v = line[2] if line[1] == "=" else line[1]

        model_dict[k] = _str_to(k, v)

    return model_dict


# ==================================================
def win_reader(topdir, seedname, encoding="UTF-8"):
    """
    read seedname.win file and return
        - kpoints: k points.
        - kpoint: representative k points.
        - kpoint_path: high-symmetry line in k space.
        - unit_cell_cart: transform matrix, [a1,a2,a3].
        - atoms_frac: atomic positions in fractional coordinates with respect to the lattice vectors, {atom: [r1,r2,r3]}.
        - atoms_cart: atomic positions in cartesian coordinates, {atom: [rx,ry,rz]}.

    Args:
        topdir (str): directory of seedname.win file.
        seedname (str): seedname.
        encoding (str, optional): encoding.

    Returns:
        dict: dictionary form of seedname.win.
    """
    topdir = topdir[:-1] if topdir[-1] == "/" else topdir
    f = open(topdir + "/" + seedname + ".win", "r", encoding=encoding)
    data = f.readlines()
    f.close()

    data_lower = [v.lower().replace("\n", "") for v in data]

    # kpoints
    if "begin kpoints" in data_lower and "end kpoints" in data_lower:
        kpoints = data[data_lower.index("begin kpoints") + 1 : data_lower.index("end kpoints")]
        kpoints = [[vi for vi in v.replace("\n", "").split() if vi != ""] for v in kpoints]
        kpoints = np.array([[float(ki) for ki in k] for k in kpoints])
        kpoints = np.mod(kpoints, 1)  # 0 <= kj < 1.0
    else:
        kpoints = None

    # kpoint, kpoint_path
    if "begin kpoint_path" in data_lower and "end kpoint_path" in data_lower:
        k_data = data[data_lower.index("begin kpoint_path") + 1 : data_lower.index("end kpoint_path")]
        k_data = [[vi for vi in v.replace("\n", "").split() if vi != ""] for v in k_data]
        kpoint = {}
        kpoint_path = ""
        cnt = 1
        for X, ki1, ki2, ki3, Y, kf1, kf2, kf3 in k_data:
            if cnt == 1:
                kpoint_path += f"{X}-{Y}-"
            else:
                if kpoint_path.split("-")[-2] == X:
                    kpoint_path += f"{Y}-"
                else:
                    kpoint_path = kpoint_path[:-1]
                    kpoint_path += f"|{X}-{Y}-"
            if X not in kpoint:
                kpoint[X] = np.array([float(ki1), float(ki2), float(ki3)])
            if Y not in kpoint:
                kpoint[Y] = np.array([float(kf1), float(kf2), float(kf3)])

            cnt += 1

        kpoint_path = kpoint_path[:-1]
    else:
        kpoint = None
        kpoint_path = None

    # unit_cell_cart
    if "begin unit_cell_cart" in data_lower and "end unit_cell_cart" in data_lower:
        units = data[data_lower.index("begin unit_cell_cart") + 1]
        units = units.replace(" ", "").replace("\n", "").lower()
        n = 2 if units in ("bohr", "ang") else 1

        unit_cell_cart_data = data[
            data_lower.index("begin unit_cell_cart") + n : data_lower.index("end unit_cell_cart")
        ]
        unit_cell_cart_data = [[vi for vi in v.replace("\n", "").split() if vi != ""] for v in unit_cell_cart_data]
        unit_cell_cart = np.array([[float(ri) for ri in r] for r in unit_cell_cart_data])

        if units == "bohr":
            unit_cell_cart *= 0.529177249
    else:
        units = None
        unit_cell_cart = None

    # atoms_frac
    if "begin atoms_frac" in data_lower and "end atoms_frac" in data_lower:
        ap_data = data[data_lower.index("begin atoms_frac") + 1 : data_lower.index("end atoms_frac")]
        ap_data = [[vi for vi in v.replace("\n", "").split() if vi != ""] for v in ap_data]
        atoms_frac = {X: np.array([float(r1), float(r2), float(r3)]) for X, r1, r2, r3 in ap_data}
    else:
        atoms_frac = None

    # atoms_cart
    if "begin atoms_cart" in data_lower and "end atoms_cart" in data_lower:
        units = data[data_lower.index("begin unit_cell_cart") + 1]
        units = units.replace(" ", "").replace("\n", "").lower()
        n = 2 if units in ("bohr", "ang") else 1

        ap_data = data[data_lower.index("begin atoms_cart") + n : data_lower.index("end atoms_cart")]
        ap_data = [[vi for vi in v.replace("\n", "").split() if vi != ""] for v in ap_data]
        atoms_cart = {X: np.array([float(r1), float(r2), float(r3)]) for X, r1, r2, r3 in ap_data}

        if units == "bohr":
            atoms_cart = {X: v * 0.529177249 for X, v in atoms_cart.items()}
    else:
        atoms_cart = None

    return kpoints, kpoint, kpoint_path, unit_cell_cart, atoms_frac, atoms_cart


# ==================================================
def amn_reader(topdir, seedname, encoding="UTF-8"):
    """
    read seedname.amn file which is obtained by plane-wave DFT calculation software, such as QuantumEspresso and VASP.
    seedname.amn file includes overlap matrix elements between Kohn-Sham orbitals (KSOs) [ψ^{KS}_{m}(k)] and pseudo atomic (PAOs) orbitals [φ^{A}_{n}(k)] at each k point, A_{mn}(k) = <ψ^{KS}_{m}(k)|φ^{A}_{n}(k)>.

    Args:
        topdir (str): directory of seedname.amn file.
        seedname (str): seedname.
        encoding (str, optional): encoding.

    Returns:
        ndarray: overlap matrix elements between Kohn-Sham orbitals [ψ_{m}(k)] and pseudo atomic orbitals [φ_{n}(k)] at each k point, A_{mn}(k) = <ψ_{m}(k)|φ_{n}(k)>.

    Notes:
        - m: Kohn-Sham orbital # starting from 1.
        - n: pseudo atomic orbital # starting from 1.
    """
    f = open(topdir + "/" + seedname + ".amn", "r", encoding=encoding)
    amn_data = f.readlines()
    f.close()

    num_bands, num_k, num_wann = [int(x) for x in amn_data[1].split()]
    amn_data = np.genfromtxt(amn_data[2:]).reshape(num_k, num_wann, num_bands, 5)
    Aks = np.transpose(amn_data[:, :, :, 3] + 1j * amn_data[:, :, :, 4], axes=(0, 2, 1))

    return Aks


# ==================================================
def eig_reader(topdir, seedname, encoding="UTF-8"):
    """
    read seedname.eig file which is obtained by plane-wave DFT calculation software, such as QuantumEspresso and VASP.
    seedname.eig file includes Kohn-Sham energy at each k point, E_{m}(k).

    Args:
        topdir (str): directory of seedname.eig file.
        seedname (str): seedname.
        encoding (str, optional): encoding.

    Returns:
        ndarray: Kohn-Sham energy at each k point, E_{m}(k).

    Notes:
        - m: Kohn-Sham orbital # starting from 1.
    """
    f = open(topdir + "/" + seedname + ".eig", "r", encoding=encoding)
    eig_data = f.readlines()
    f.close()

    eig_data = [[v for v in lst.rstrip("\n").split(" ") if v != ""] for lst in eig_data]
    eig_data = [[float(v) if "." in v else int(v) for v in lst] for lst in eig_data]

    num_bands, num_k = np.max([v[0] for v in eig_data]), np.max([v[1] for v in eig_data])

    Eks = np.array([[eig_data[k * num_bands + m][2] for m in range(num_bands)] for k in range(num_k)])

    return Eks


# ==================================================
def nnkp_reader(topdir, seedname, encoding="UTF-8"):
    """
    read seedname.nnkp file which is obtained by plane-wave DFT calculation software, such as QuantumEspresso and VASP.
    seedname.nnkp file includes k point values in the reduced coorinate, [[k1 k2 k3]] (0 <= kj <= 1, j = 1,2,3).

    Args:
        topdir (str): directory of seedname.nnkp file.
        seedname (str): seedname.
        encoding (str, optional): encoding.

    Returns:
        ndarray: k points (reduced coodinate).
    """
    f = open(topdir + "/" + seedname + ".nnkp", "r", encoding=encoding)
    nnkp_data = f.readlines()
    f.close()

    nnkp_data = nnkp_data[nnkp_data.index("begin kpoints\n") + 2 : nnkp_data.index("end kpoints")]
    nnkp_data = [[vi for vi in v.replace("\n", "").split() if vi != ""] for v in nnkp_data]

    k_points = np.array([[float(ki) for ki in k] for k in nnkp_data])

    return k_points


# ==================================================
def atomic_proj_reader(savedir, outdir="."):
    # identify states from projwfc.x's stdout
    p = os.popen("grep -a -n Giannozzi " + outdir + "/pdos.out | tail -1", "r")
    n = p.readline().split()[0].strip(":").strip()
    p.close()
    p = os.popen("tail -n +" + n + " " + outdir + '/pdos.out | grep "state #"', "r")

    states = []
    for x in p.readlines():
        y = x.split("atom")[1]
        iatom = int(y.split()[0]) - 1
        z = y.replace(")\n", "").split("=")
        if y.find("m_j") < 0:
            l = int(z[1].replace("m", ""))
            m = int(z[2])
            states.append([iatom, l, m])
        else:
            j = float(z[1].replace("l", ""))
            l = int(z[2].replace("m_j", ""))
            mj = float(z[3])
            states.append([iatom, j, l, mj])
    p.close()

    # read in projections from atomic_proj.xml
    f = open(savedir + "/atomic_proj.xml", "r")
    atomic_proj_data = xmltodict.parse(f.read())
    nbnd = int(atomic_proj_data["PROJECTIONS"]["HEADER"]["@NUMBER_OF_BANDS"])
    nkp = int(atomic_proj_data["PROJECTIONS"]["HEADER"]["@NUMBER_OF_K-POINTS"])
    # spinpol = int(atomic_proj_data["PROJECTIONS"]["HEADER"]["@NUMBER_OF_SPIN_COMPONENTS"]) == 2
    natomwfc = int(atomic_proj_data["PROJECTIONS"]["HEADER"]["@NUMBER_OF_ATOMIC_WFC"])

    proj_data = atomic_proj_data["PROJECTIONS"]["EIGENSTATES"]["PROJS"]

    if len(proj_data) == 0:
        raise RuntimeError("no projections found")

    projections = np.zeros((nkp, natomwfc, nbnd), np.complex128)
    for k in range(nkp):
        for a in range(natomwfc):
            lst = proj_data[k]["ATOMIC_WFC"][a]["#text"].split("\n")
            for i in range(nbnd):
                v = [vi for vi in lst[i].split(" ") if vi != ""]
                re, im = float(v[0]), float(v[1])
                projections[k, a, i] = re + 1j * im

    projections = projections.transpose(0, 2, 1).conjugate()

    # read in overlaps from atomic_proj.xml
    overlaps = []
    for k in range(nkp):
        lst = atomic_proj_data["PROJECTIONS"]["OVERLAPS"]["OVPS"][k]["#text"].split("\n")
        m = []
        for i in range(natomwfc * natomwfc):
            v = [vi for vi in lst[i].split(" ") if vi != ""]
            re, im = float(v[0]), float(v[1])
            m.append(re + 1j * im)

        m = np.reshape(np.array(m), (8, 8))
        overlaps.append(m)

    overlaps = np.array(overlaps)

    # read in eigenvalues from atomic_proj.xml
    ry_to_ev = 13.605684958731
    eigenvalues = []
    for k in range(nkp):
        lst = atomic_proj_data["PROJECTIONS"]["EIGENSTATES"]["E"][k].split("\n")
        lst = [ry_to_ev * float(vi) for v in lst for vi in v.split(" ") if vi != ""]
        eigenvalues.append(lst)

    eigenvalues = np.array(eigenvalues)

    # read in k-points from atomic_proj.xml
    k_points = []
    for k in range(nkp):
        lst = atomic_proj_data["PROJECTIONS"]["EIGENSTATES"]["K-POINT"][k]["#text"].split("\n")
        lst = [float(vi) for v in lst for vi in v.split(" ") if vi != ""]
        k_points.append(lst)

    k_points = np.array(k_points)

    return states, projections, overlaps, eigenvalues, k_points
