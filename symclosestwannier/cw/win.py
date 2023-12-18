"""
Win manages input file for wannier90.x, seedname.win file.
"""
import os
import numpy as np


_default = {
    "seedname": "cwannier",
    "num_k": 1,
    "num_bands": 1,
    "num_wann": 1,
    #
    "dis_num_iter": 0,
    "num_iter": 200,
    "dis_froz_max": +100000,
    "dis_froz_min": -100000,
    "dis_win_max": +100000,
    "dis_win_min": -100000,
    "dis_mix_ratio": 0.5,
    #
    "mp_grid": [1, 1, 1],
    "kpoints": [[0, 0, 0]],
    "kpoint": None,
    "kpoint_path": None,
    "unit_cell_cart": None,
    "atoms_frac": None,
    "atoms_cart": None,
    #
    "kmesh": [1, 1, 1],
    "kmesh_spacing": [1, 1, 1],
    "adpt_smr": True,
    "adpt_smr_fac": np.sqrt(2),
    "adpt_smr_max": 1.0,
    "smr_type": "gauss",
    "smr_fixed_en_width": 0.0,
    "spin_decomp": False,
    # berry
    "berry": False,
    "berry_task": "",
    "berry_kmesh": [1, 1, 1],
    "berry_kmesh_spacing": [1, 1, 1],
    # kubo
    "kubo_freq_max": None,
    "kubo_freq_min": 0.0,
    "kubo_freq_step": 0.01,
    "kubo_eigval_max": None,
    "kubo_adpt_smr": True,
    "kubo_adpt_smr_fac": np.sqrt(2),
    "kubo_adpt_smr_max": 1.0,
    "kubo_smr_fixed_en_width": 0.0,
    "kubo_smr_type": "gauss",
    # gyrotropic
    "gyrotropic": False,
    # boltzwann
    "boltzwann": False,
}


# ==================================================
class Win(dict):
    """
    Win manages input file for wannier90.x, seedname.win file.

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
    """

    # ==================================================
    def __init__(self, topdir=None, seedname="cwannier", dic=None):
        """
        initialize the class.

        Args:
            topdir (str, optional): directory of seedname.win file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of Win.
        """
        super().__init__()

        self._topdir = topdir
        self._seedname = seedname

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "win"))
            self.update(self.read(file_name))
            self["seedname"] = seedname
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name="cwannier.win"):
        """
        read seedname.win file.

        Args:
            file_name (str, optional): file name.

        Returns:
            dict: dictionary form of seedname.win.
                - num_k               : # of k points (int), [1].
                - num_bands           : # of bands passed to the code (int), [1].
                - num_wann            : # of WFs (int), [1].
                - kpoint*             : representative k points, [1].
                - kpoint_path*        : high-symmetry line in k-space, [None].
                - unit_cell_cart*     : transform matrix, [a1,a2,a3], [None].
                - atoms_frac*         : atomic positions in fractional coordinates with respect to the lattice vectors, {atom: [r1,r2,r3]} [None].
                - atoms_cart*         : atomic positions in cartesian coordinates, {atom: [rx,ry,rz]} [None].
                - dis_num_iter*       : # of iterations for disentanglement (int), [0].
                - num_iter*           : # of iterations for maximal localization (int), [200].
                - dis_froz_max        : top of the inner (frozen) energy window (float), [+100000].
                - dis_froz_min        : bottom of the inner (frozen) energy window (float), [-100000].
                - dis_win_max         : top of the outer energy window (float), [+100000].
                - dis_win_min         : bottom of the outer energy window (float), [-100000].
                - dis_mix_ratio       : mixing ratio during the disentanglement (float), [0.5].
                - mp_grid             : dimensions of the Monkhorst-Pack grid of k-points (list), [[1, 1, 1]],
                - kpoints             : k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
                - kpoint              : representative k points (dict), [None].
                - kpoint_path         : k-points along high symmetry line in Brillouin zonen, [[k1, k2, k3]] (crystal coordinate) (str), [None].
                - unit_cell_cart      : transform matrix, [a1,a2,a3] (list), [None].

            # only used for postcw calculation (same as postw90).
                - kmesh               : dimensions of the Monkhorst-Pack grid of k-points for response calculation (list), [[1, 1, 1]].
                - kmesh_spacing       : minimum distance for neighboring k points along each of the three directions in k space (The units are [Ang])(list), [1,1,1]].
                - adpt_smr            : Determines whether to use an adaptive scheme for broadening the DOS and similar quantities defined on the energy axis (bool), [True].
                - adpt_smr_fac        : The width ηnk of the broadened delta function used to determine the contribution to the spectral property (DOS, ...) from band n at point k (float), [sqrt(2)].
                - adpt_smr_max        : Maximum allowed value for the adaptive energy smearing [eV] (float), [1.0].
                - smr_type            : Defines the analytical form used for the broadened delta function in the computation of the DOS and similar quantities defined on the energy axis, gauss/m-pN/m-v or cold/f-d (str), [gauss].
                - smr_fixed_en_width  : Energy width for the smearing function for the DOS. Used only if adpt_smr is false (The units are [eV]) (flaot), [0.0].
                - spin_decomp         : If true, extra columns are added to some output files (such as seedname-dos.dat for the dos module, and analogously for the berry and BoltzWann modules) (bool), [False].
            # berry
                - berry               : Determines whether to enter the berry routines (bool), [False].
                - berry_task          : The quantity to compute when berry=true, ahc/morb/kubo/sc/shc/kdotp (str).
                - berry_kmesh         : Overrides the kmesh global variable.
                - berry_kmesh_spacing : Overrides the kmesh_spacing global variable.
            # kubo
                - kubo_freq_max       : Upper limit of the frequency range for computing the optical conductivity, JDOS and ac SHC. (The units are [eV]) (float), [If an inner energy window was specified, the default value is dis_froz_max-fermi_energy+0.6667. Otherwise it is the difference between the maximum and the minimum energy eigenvalue stored in seedname.eig, plus 0.6667.].
                - kubo_freq_min       : Lower limit of the frequency range for computing the optical conductivity, JDOS and ac SHC. (The units are [eV]) (float), [0.0].
                - kubo_freq_step      : Difference between consecutive values of the optical frequency between kubo_freq_min and kubo_freq_max. (The units are [eV]) (float), [0.01].
                - kubo_eigval_max     : Maximum energy eigenvalue of the eigenstates to be included in the evaluation of the optical conductivity, JDOS and ac SHC. (The units are [eV]) (float), [If an inner energy window was specified, the default value is the upper bound of the inner energy window plus 0.6667. Otherwise it is the maximum energy eigenvalue stored in seedname.eig plus 0.6667.].
                - kubo_adpt_smr       : Overrides the adpt_smr global variable.
                - kubo_adpt_smr_fac   : Overrides the adpt_smr_fac global variable.
                - kubo_adpt_smr_max   : Overrides the adpt_smr_max global variable.
                - kubo_smr_fixed_en_width : Overrides the smr_fixed_en_width global variable.
                - kubo_smr_type       : Overrides the smr_type global variable.
            # ahc
            # morb
            # shc
            # gyrotropic
                - gyrotropic          : Determines whether to enter the gyrotropic routines (bool), [False].
            # boltzwann
                - boltzwann           : Determines whether to enter the boltzwann routines (bool), [False].
        """
        if os.path.exists(file_name):
            with open(file_name) as fp:
                win_data = fp.readlines()
        else:
            raise Exception("failed to read win file: " + file_name)

        d = Win._default().copy()

        win_data = [v.replace("\n", "") for v in win_data]
        win_data_lower = [v.lower().replace("\n", "") for v in win_data]

        d["num_bands"] = self._get_param_keyword(win_data, "num_bands", 0, dtype=int)
        d["num_wann"] = self._get_param_keyword(win_data, "num_wann", 0, dtype=int)
        mp_grid = self._get_param_keyword(win_data, "mp_grid", "1  1  1", dtype=str)
        d["mp_grid"] = [int(x) for x in mp_grid.split()]
        d["num_k"] = np.prod(d["mp_grid"])

        if d["num_bands"] > d["num_wann"]:
            d["dis_num_iter"] = self._get_param_keyword(win_data, "dis_num_iter", 200, dtype=int)
        else:
            d["dis_num_iter"] = 0
        d["num_iter"] = self._get_param_keyword(win_data, "num_iter", 200, dtype=int)
        d["dis_froz_max"] = self._get_param_keyword(win_data, "dis_froz_max", +100000, dtype=float)
        d["dis_froz_min"] = self._get_param_keyword(win_data, "dis_froz_min", -100000, dtype=float)
        d["dis_win_max"] = self._get_param_keyword(win_data, "dis_win_max", +100000, dtype=float)
        d["dis_win_min"] = self._get_param_keyword(win_data, "dis_win_min", -100000, dtype=float)
        d["dis_mix_ratio"] = self._get_param_keyword(win_data, "dis_mix_ratio", 0.5, dtype=float)

        for i, line in enumerate(win_data_lower):
            if "begin kpoints" in line:
                kpoints = np.genfromtxt(win_data[i + 1 : i + 1 + d["num_k"]], dtype=float)
                kpoints = np.mod(kpoints, 1)  # 0 <= kj < 1.0
                if kpoints.ndim == 1:
                    d["kpoints"] = [kpoints.tolist()]
                else:
                    d["kpoints"] = kpoints.tolist()

            if "begin kpoint_path" in line:
                k_data = win_data[i + 1 : win_data_lower.index("end kpoint_path")]
                k_data = [[vi for vi in v.split()] for v in k_data]
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
                        kpoint[X] = [float(ki1), float(ki2), float(ki3)]
                    if Y not in kpoint:
                        kpoint[Y] = [float(kf1), float(kf2), float(kf3)]

                    cnt += 1

                kpoint_path = kpoint_path[:-1]

                d["kpoint"] = kpoint
                d["kpoint_path"] = kpoint_path

            if "begin unit_cell_cart" in line:
                units = win_data[i + 1]
                units = units.replace(" ", "").lower()
                n = 2 if units in ("bohr", "ang") else 1
                unit_cell_cart_data = win_data[i + n : win_data_lower.index("end unit_cell_cart")]
                unit_cell_cart = np.array([[float(vi) for vi in v.split() if vi != ""] for v in unit_cell_cart_data])
                if units == "bohr":
                    unit_cell_cart *= 0.529177249

                d["units"] = units
                d["unit_cell_cart"] = unit_cell_cart.tolist()

            if "begin atoms_frac" in line:
                ap_data = win_data[i + 1 : win_data_lower.index("end atoms_frac")]
                ap_data = [[vi for vi in v.split() if vi != ""] for v in ap_data]

                atoms_frac = {}
                cnt_X = {}
                for X, r1, r2, r3 in ap_data:
                    if X not in cnt_X:
                        atoms_frac[(X, 1)] = [float(r1), float(r2), float(r3)]
                        cnt_X[X] = 1
                    else:
                        atoms_frac[(X, cnt_X[X] + 1)] = [float(r1), float(r2), float(r3)]
                        cnt_X[X] += 1

                d["atoms_frac"] = atoms_frac

            if "begin atoms_cart" in line:
                units = d["units"]
                n = 2 if units in ("bohr", "ang") else 1
                ap_data = win_data[i + n : win_data_lower.index("end atoms_cart")]
                ap_data = [[vi for vi in v.split() if vi != ""] for v in ap_data]

                atoms_cart = {}
                cnt_X = {}
                for X, r1, r2, r3 in ap_data:
                    if X not in cnt_X:
                        atoms_cart[(X, 1)] = [float(r1), float(r2), float(r3)]
                        cnt_X[X] = 1
                    else:
                        atoms_cart[(X, cnt_X[X] + 1)] = [float(r1), float(r2), float(r3)]
                        cnt_X[X] += 1

                if units == "bohr":
                    atoms_cart = {k: v * 0.529177249 for k, v in atoms_cart.items()}

                d["atoms_cart"] = atoms_cart

        del d["units"]

        if d["atoms_cart"] is not None:
            if d["atoms_frac"] is None:
                A = d["unit_cell_cart"]
                d["atoms_frac"] = {k: (np.linalg.inv(A) @ np.array(v)).tolist() for k, v in d["atoms_cart"].items()}

        if d["atoms_frac"] is not None:
            if d["atoms_cart"] is None:
                A = d["unit_cell_cart"]
                d["atoms_cart"] = {k: (np.array(v) @ np.array(A)).tolist() for k, v in d["atoms_frac"].items()}

        # postcw
        kmesh = self._get_param_keyword(win_data, "kmesh", "1  1  1", dtype=str)
        d["kmesh"] = [int(x) for x in kmesh.split()]
        kmesh_spacing = self._get_param_keyword(win_data, "kmesh_spacing", "1  1  1", dtype=str)
        d["kmesh_spacing"] = [int(x) for x in kmesh_spacing.split()]
        d["adpt_smr"] = self._get_param_keyword(win_data, "adpt_smr", True, dtype=bool)
        d["adpt_smr_fac"] = self._get_param_keyword(win_data, "adpt_smr_fac", np.sqrt(2), dtype=float)
        d["adpt_smr_max"] = self._get_param_keyword(win_data, "adpt_smr_max", 1.0, dtype=float)
        d["smr_type"] = self._get_param_keyword(win_data, "smr_type", "gauss", dtype=str).replace(" ", "")
        d["smr_fixed_en_width"] = self._get_param_keyword(win_data, "smr_fixed_en_width", 0.0, dtype=float)
        d["spin_decomp"] = self._get_param_keyword(win_data, "spin_decomp", False, dtype=bool)

        d["berry"] = self._get_param_keyword(win_data, "berry", False, dtype=bool)
        berry_task = self._get_param_keyword(win_data, "berry_task", None, dtype=str)
        if berry_task is not None:
            berry_task = berry_task.replace(" ", "")
        d["berry_task"] = berry_task
        berry_kmesh = self._get_param_keyword(win_data, "berry_kmesh", "1  1  1", dtype=str)
        d["berry_kmesh"] = [int(x) for x in berry_kmesh.split()]
        berry_kmesh_spacing = self._get_param_keyword(win_data, "berry_kmesh_spacing", "1  1  1", dtype=str)
        d["berry_kmesh_spacing"] = [int(x) for x in berry_kmesh_spacing.split()]

        d["kubo_freq_max"] = self._get_param_keyword(win_data, "kubo_freq_max", None, dtype=float)
        d["kubo_freq_min"] = self._get_param_keyword(win_data, "kubo_freq_min", 0.0, dtype=float)
        d["kubo_freq_step"] = self._get_param_keyword(win_data, "kubo_freq_step", 0.01, dtype=float)
        d["kubo_eigval_max"] = self._get_param_keyword(win_data, "kubo_eigval_max", None, dtype=float)
        d["kubo_adpt_smr"] = self._get_param_keyword(win_data, "kubo_adpt_smr", True, dtype=bool)
        d["kubo_adpt_smr_fac"] = self._get_param_keyword(win_data, "kubo_adpt_smr_fac", np.sqrt(2), dtype=float)
        d["kubo_adpt_smr_max"] = self._get_param_keyword(win_data, "kubo_adpt_smr_max", 1.0, dtype=float)
        d["kubo_smr_type"] = self._get_param_keyword(win_data, "smr_type", "gauss", dtype=str).replace(" ", "")
        d["kubo_smr_fixed_en_width"] = self._get_param_keyword(win_data, "kubo_smr_fixed_en_width", 0.0, dtype=float)

        d["gyrotropic"] = self._get_param_keyword(win_data, "gyrotropic", False, dtype=bool)
        d["boltzwann"] = self._get_param_keyword(win_data, "boltzwann", False, dtype=bool)

        return d

    # ==================================================
    def _get_param_keyword(self, lines, keyword, default_value=None, dtype=int):
        data = None
        keys = []
        for line in lines:
            line = line.replace("\n", "")
            if line.startswith(keyword):
                if len(line.split("=")) > 1:
                    key, data = line.split("=")
                    key = key.replace(" ", "")
                    if key == keyword:
                        data = data.split("!")[0]
                    assert key not in keys, key + " is defined more than once"
                elif len(line.split(":")) > 1:
                    key, data = line.split(":")
                    key = key.replace(" ", "")
                    if key == keyword:
                        data = data.split("!")[0]
                    assert key not in keys, key + " is defined more than once"

        if data is None:
            data = default_value
        else:
            if data.lower() in ("true", ".true."):
                data = True
            elif data.lower() in ("false", ".false."):
                data = False

        if data is None:
            return None

        return dtype(data)

    # ==================================================
    @classmethod
    def _default(cls):
        return _default
