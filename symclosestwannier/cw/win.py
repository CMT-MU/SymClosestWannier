"""
Win manages input file for wannier90.x, seedname.win file.
"""

import os
import numpy as np

from gcoreutils.nsarray import NSArray


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
    "exclude_bands": None,
    #
    "mp_grid": [1, 1, 1],
    "kpoints": [[0, 0, 0]],
    "kpoint": None,
    "kpoint_path": None,
    "unit_cell_cart": None,
    "atoms_frac": None,
    "atoms_cart": None,
    "spinors": False,
    "spin_moment": False,
    #
    "kmesh": [1, 1, 1],
    "kmesh_spacing": [1, 1, 1],
    "adpt_smr": False,
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
    # berry curvature, ahc, shc
    "berry_curv_unit": "ang2",
    "berry_curv_adpt_kmesh": 1,
    "berry_curv_adpt_kmesh_thresh": 100,
    "fermi_energy": 0.0,
    "fermi_energy_max": None,
    "fermi_energy_min": None,
    "fermi_energy_step": 0.01,
    "fermi_energy_list": None,
    "num_fermi": 0,
    "shc_freq_scan": False,
    "shc_alpha": 1,
    "shc_beta": 2,
    "shc_gamma": 3,
    "shc_bandshift": False,
    "shc_bandshift_firstband": None,
    "shc_bandshift_energyshift": 0.0,
    # kubo
    "kubo_freq_max": None,
    "kubo_freq_min": 0.0,
    "kubo_freq_step": 0.01,
    "kubo_eigval_max": +100000,
    "kubo_adpt_smr": False,
    "kubo_adpt_smr_fac": np.sqrt(2),
    "kubo_adpt_smr_max": 1.0,
    "kubo_smr_fixed_en_width": 0.0,
    "kubo_smr_type": "gauss",
    # gyrotropic
    "gyrotropic": False,
    "gyrotropic_task": "",
    "gyrotropic_kmesh": [1, 1, 1],
    "gyrotropic_kmesh_spacing": [1, 1, 1],
    "gyrotropic_freq_max": None,
    "gyrotropic_freq_min": 0.0,
    "gyrotropic_freq_step": 0.01,
    "gyrotropic_eigval_max": +100000,
    "gyrotropic_degen_thresh": 0.0,
    "gyrotropic_smr_fixed_en_width": 0.0,
    "gyrotropic_smr_type": "gauss",
    "gyrotropic_band_list": None,
    "gyrotropic_box_center": [0.5, 0.5, 0.5],
    "gyrotropic_box_b1": [1.0, 0.0, 0.0],
    "gyrotropic_box_b2": [0.0, 1.0, 0.0],
    "gyrotropic_box_b3": [0.0, 0.0, 1.0],
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
        Win manages input file for wannier90.x, seedname.win file.

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
                - num_k           : # of k points (int), [1].
                - num_bands       : # of bands passed to the code (int), [1].
                - num_wann        : # of WFs (int), [1].
                - kpoint*         : representative k points, [1].
                - kpoint_path*    : high-symmetry line in k-space, [None].
                - unit_cell_cart* : transform matrix, [a1,a2,a3], [None].
                - atoms_frac*     : atomic positions in fractional coordinates with respect to the lattice vectors, {atom: [r1,r2,r3]} [None].
                - atoms_cart*     : atomic positions in cartesian coordinates, {atom: [rx,ry,rz]} [None].
                - spinors         : WFs are spinors? (bool) [False].
                - spin_moment     : Determines whether to evaluate the spin moment (bool), [False].
                - dis_num_iter*   : # of iterations for disentanglement (int), [0].
                - num_iter*       : # of iterations for maximal localization (int), [200].
                - dis_froz_max    : top of the inner (frozen) energy window (float), [+100000].
                - dis_froz_min    : bottom of the inner (frozen) energy window (float), [-100000].
                - dis_win_max     : top of the outer energy window (float), [+100000].
                - dis_win_min     : bottom of the outer energy window (float), [-100000].
                - dis_mix_ratio   : mixing ratio during the disentanglement (float), [0.5].
                - mp_grid         : dimensions of the Monkhorst-Pack grid of k-points (list), [[1, 1, 1]],
                - kpoints         : k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
                - kpoint          : representative k points (dict), [None].
                - kpoint_path     : k-points along high symmetry line in Brillouin zonen, [[k1, k2, k3]] (crystal coordinate) (str), [None].
                - unit_cell_cart  : transform matrix, [a1,a2,a3] (list), [None].

            # only used for postcw calculation (same as postw90).
                - kmesh                        : dimensions of the Monkhorst-Pack grid of k-points for response calculation (list), [[1, 1, 1]].
                - kmesh_spacing                : minimum distance for neighboring k points along each of the three directions in k space (The units are [Ang])(list), [1,1,1]].
                - adpt_smr                     : Determines whether to use an adaptive scheme for broadening the DOS and similar quantities defined on the energy axis (bool), [True].
                - adpt_smr_fac                 : The width ηnk of the broadened delta function used to determine the contribution to the spectral property (DOS, ...) from band n at point k (float), [sqrt(2)].
                - adpt_smr_max                 : Maximum allowed value for the adaptive energy smearing [eV] (float), [1.0].
                - smr_type                     : Defines the analytical form used for the broadened delta function in the computation of the DOS and similar quantities defined on the energy axis, gauss/m-pN/m-v or cold/f-d (str), [gauss].
                - smr_fixed_en_width           : Energy width for the smearing function for the DOS. Used only if adpt_smr is false (The units are [eV]) (flaot), [0.0].
                - spin_decomp                  : If true, extra columns are added to some output files (such as seedname-dos.dat for the dos module, and analogously for the berry and BoltzWann modules) (bool), [False].

            # berry
                - berry                        : Determines whether to enter the berry routines (bool), [False].
                - berry_task                   : The quantity to compute when berry=true, ahc/morb/kubo/sc/shc/kdotp/me (str).
                - berry_kmesh                  : Overrides the kmesh global variable.
                - berry_kmesh_spacing          : Overrides the kmesh_spacing global variable.

            # berry curvature, ahc, shc
                - berry_curv_unit              : Unit of Berry curvature, ang2/bohr2, ['ang2'].
                - berry_curv_adpt_kmesh        : Linear dimension of the adaptively refined k-mesh used to compute the anomalous/spin Hall conductivity, [1].
                - berry_curv_adpt_kmesh_thresh : Threshold magnitude of the Berry curvature for adaptive refinement, [100].
                - fermi_energy                 : fermi energy (float), [0.0].
                - fermi_energy_max             : Upper limit of the Fermi energy range (float), [None].
                - fermi_energy_min             : Lower limit of the Fermi energy range (float), [None].
                - fermi_energy_step            : Step for increasing the Fermi energy in the specified range. (The units are [eV]) (float), [0.01].
                - fermi_energy_list            : list of fermi energy (list), [None].
                - num_fermi                    : number of fermi energies (int), [0].
                - shc_freq_scan                : Determines whether to calculate the frequency scan (i.e. the ac SHC) or the Fermi energy scan (i.e. the dc SHC) of the spin Hall conductivity. The default value is false, which means dc SHC is calculated. (bool), [False].
                - shc_alpha                    : The α index of spin Hall conductivity σ^{spin γ}_{αβ}, i.e. the direction of spin current. Possible values are 1, 2 and 3, representing the x, y and z directions respectively. (int), [1].
                - shc_beta                     : The β index of spin Hall conductivity σ^{spin γ}_{αβ}, i.e. the direction of applied electric field. Possible values are 1, 2 and 3, representing the x, y and z directions respectively. (int), [2].
                - shc_gamma                    : The γ index of spin Hall conductivity σ^{spin γ}_{αβ}, i.e. the spin direction of spin current. Possible values are 1, 2 and 3, representing the x, y and z directions respectively. (int), [3].
                - shc_bandshift                : Shift all conduction bands by a given amount (float), [False].
                - shc_bandshift_firstband      : Index of the first band to shift (int), [None].
                - shc_bandshift_energyshift    : Energy shift of the conduction bands (eV) (float), [0.0].

            # kubo
                - kubo_freq_max           : Upper limit of the frequency range for computing the optical conductivity, JDOS and ac SHC. (The units are [eV]) (float), [If an inner energy window was specified, the default value is dis_froz_max-fermi_energy+0.6667. Otherwise it is the difference between the maximum and the minimum energy eigenvalue stored in seedname.eig, plus 0.6667.].
                - kubo_freq_min           : Lower limit of the frequency range for computing the optical conductivity, JDOS and ac SHC. (The units are [eV]) (float), [0.0].
                - kubo_freq_step          : Difference between consecutive values of the optical frequency between kubo_freq_min and kubo_freq_max. (The units are [eV]) (float), [0.01].
                - kubo_eigval_max         : Maximum energy eigenvalue of the eigenstates to be included in the evaluation of the optical conductivity, JDOS and ac SHC. (The units are [eV]) (float), [If an inner energy window was specified, the default value is the upper bound of the inner energy window plus 0.6667. Otherwise it is the maximum energy eigenvalue stored in seedname.eig plus 0.6667.].
                - kubo_adpt_smr           : Overrides the adpt_smr global variable (bool), [False].
                - kubo_adpt_smr_fac       : Overrides the adpt_smr_fac global variable (float), [sqrt(2)].
                - kubo_adpt_smr_max       : Overrides the adpt_smr_max global variable (float), [1.0].
                - kubo_smr_fixed_en_width : Overrides the smr_fixed_en_width global variable (float), [0.0].
                - kubo_smr_type           : Overrides the smr_type global variable (str), [gauss].

            # morb

            # gyrotropic
                - gyrotropic                    : Determines whether to enter the gyrotropic routines (bool), [False].
                - gyrotropic_task               : The quantity to compute when gyrotropic=true, -DO/-Dw/-C/-K/-spin/-NOA/-dos, (str).
                - gyrotropic_kmesh              : Overrides the kmesh global variable.
                - gyrotropic_kmesh_spacing      : Overrides the kmesh_spacing global variable.
                - gyrotropic_freq_max           : Upper limit of the frequency range for computing the optical activity. (The units are [eV]) (float), [If an inner energy window was specified, the default value is dis_froz_max-fermi_energy+0.6667. Otherwise it is the difference between the maximum and the minimum energy eigenvalue stored in seedname.eig, plus 0.6667.].
                - gyrotropic_freq_min           : Lower limit of the frequency range for computing the optical activity. (The units are [eV]) (float), [0.0].
                - gyrotropic_freq_step          : Difference between consecutive values of the optical frequency between gyrotropic_freq_min and gyrotropic_freq_max. (The units are [eV]) (float), [0.01].
                - gyrotropic_eigval_max         : Maximum energy eigenvalue of the eigenstates to be included in the evaluation of the Natural optical activity. (The units are [eV]) (float), [If an inner energy window was specified, the default value is the upper bound of the inner energy window plus 0.6667. Otherwise it is the maximum energy eigenvalue stored in seedname.eig plus 0.6667.].
                - gyrotropic_smr_max_arg        : Maximum value of smearing arg (float), [5.0].
                - gyrotropic_degen_thresh       : The threshould to eliminate degenerate bands from the calculation in order to avoid divergences. (Units are [eV]) (float), [0.0].
                - gyrotropic_smr_fixed_en_width : Overrides the smr_fixed_en_width global variable.
                - gyrotropic_smr_type           : Overrides the smr_type global variable.
                - gyrotropic_band_list          : List of bands used in the calculation.
                - gyrotropic_box_center         : Three real numbers. Optionally the integration may be restricted to a parallelogram, centered at gyrotropic_box_center and defined by vectors gyrotropic_box_b{1,2,3}.  In reduced coordinates, [0.5 0.5 0.5].
                - gyrotropic_box_b1             : Three real numbers. In reduced coordinates, [1.0 0.0 0.0].
                - gyrotropic_box_b2             : Three real numbers. In reduced coordinates, [0.0 1.0 0.0].
                - gyrotropic_box_b3             : Three real numbers. In reduced coordinates, [0.0 0.0 1.0].

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
        d["exclude_bands"] = self._get_param_keyword(win_data, "exclude_bands", None, dtype=str)

        d["spinors"] = self._get_param_keyword(win_data, "spinors", False, dtype=bool)
        d["spin_moment"] = self._get_param_keyword(win_data, "spin_moment", False, dtype=bool)

        for i, line in enumerate(win_data_lower):
            if "begin kpoints" in line:
                kpoints = np.genfromtxt(win_data[i + 1 : i + 1 + d["num_k"]], dtype=float)
                kpoints = np.mod(kpoints, 1)  # 0 <= kj < 1.0
                if kpoints.ndim == 1:
                    d["kpoints"] = [kpoints.tolist()]
                else:
                    d["kpoints"] = kpoints.tolist()

                d["kpoints"] = [kpt[:3] for kpt in d["kpoints"]]

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
                        atoms_cart[(X, 1)] = np.array([float(r1), float(r2), float(r3)])
                        cnt_X[X] = 1
                    else:
                        atoms_cart[(X, cnt_X[X] + 1)] = np.array([float(r1), float(r2), float(r3)])
                        cnt_X[X] += 1

                if units == "bohr":
                    atoms_cart = {k: v * 0.529177249 for k, v in atoms_cart.items()}

                d["atoms_cart"] = {k: v.tolist() for k, v in atoms_cart.items()}

        if "units" in d:
            del d["units"]

        if d["atoms_cart"] is not None:
            if d["atoms_frac"] is None:
                A = d["unit_cell_cart"]
                d["atoms_frac"] = {k: (np.array(v) @ np.linalg.inv(A)).tolist() for k, v in d["atoms_cart"].items()}

        if d["atoms_frac"] is not None:
            if d["atoms_cart"] is None:
                A = d["unit_cell_cart"]
                d["atoms_cart"] = {k: (np.array(v) @ np.array(A)).tolist() for k, v in d["atoms_frac"].items()}

        # shift
        dic = {k: NSArray(str(v), style="vector", fmt="sympy", real=True).shift() for k, v in d["atoms_frac"].items()}
        d["atoms_frac_shift"] = {k: [float(v[0]), float(v[1]), float(v[2])] for k, v in dic.items()}

        # postcw
        kmesh = self._get_param_keyword(win_data, "kmesh", "1  1  1", dtype=str)
        d["kmesh"] = [int(x) for x in kmesh.split()]
        kmesh_spacing = self._get_param_keyword(win_data, "kmesh_spacing", "1  1  1", dtype=str)
        d["kmesh_spacing"] = [int(x) for x in kmesh_spacing.split()]
        d["adpt_smr"] = self._get_param_keyword(win_data, "adpt_smr", False, dtype=bool)
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

        d["kubo_freq_max"] = self._get_param_keyword(win_data, "kubo_freq_max", 1.0, dtype=float)
        d["kubo_freq_min"] = self._get_param_keyword(win_data, "kubo_freq_min", 0.0, dtype=float)
        d["kubo_freq_step"] = self._get_param_keyword(win_data, "kubo_freq_step", 0.01, dtype=float)
        d["kubo_eigval_max"] = self._get_param_keyword(win_data, "kubo_eigval_max", +100000, dtype=float)
        d["kubo_adpt_smr"] = self._get_param_keyword(win_data, "kubo_adpt_smr", False, dtype=bool)
        d["kubo_adpt_smr_fac"] = self._get_param_keyword(win_data, "kubo_adpt_smr_fac", np.sqrt(2), dtype=float)
        d["kubo_adpt_smr_max"] = self._get_param_keyword(win_data, "kubo_adpt_smr_max", 1.0, dtype=float)
        d["kubo_smr_type"] = self._get_param_keyword(win_data, "smr_type", "gauss", dtype=str).replace(" ", "")
        d["kubo_smr_fixed_en_width"] = self._get_param_keyword(win_data, "kubo_smr_fixed_en_width", 0.0, dtype=float)

        d["gyrotropic"] = self._get_param_keyword(win_data, "gyrotropic", False, dtype=bool)
        gyrotropic_task = self._get_param_keyword(win_data, "gyrotropic_task", None, dtype=str)
        if gyrotropic_task is not None:
            d["gyrotropic_task"] = gyrotropic_task.replace(" ", "")
        else:
            d["gyrotropic_task"] = None

        gyrotropic_kmesh = self._get_param_keyword(win_data, "gyrotropic_kmesh", "1  1  1", dtype=str)
        d["gyrotropic_kmesh"] = [int(x) for x in gyrotropic_kmesh.split()]
        gyrotropic_kmesh_spacing = self._get_param_keyword(win_data, "gyrotropic_kmesh_spacing", "1  1  1", dtype=str)
        d["gyrotropic_kmesh_spacing"] = [int(x) for x in gyrotropic_kmesh_spacing.split()]

        d["gyrotropic_freq_max"] = self._get_param_keyword(win_data, "gyrotropic_freq_max", 1.0, dtype=float)
        d["gyrotropic_freq_min"] = self._get_param_keyword(win_data, "gyrotropic_freq_min", 0.0, dtype=float)
        d["gyrotropic_freq_step"] = self._get_param_keyword(win_data, "gyrotropic_freq_step", 0.01, dtype=float)
        d["gyrotropic_eigval_max"] = self._get_param_keyword(win_data, "gyrotropic_eigval_max", +100000, dtype=float)
        d["gyrotropic_smr_type"] = self._get_param_keyword(win_data, "smr_type", "gauss", dtype=str).replace(" ", "")
        d["gyrotropic_smr_fixed_en_width"] = self._get_param_keyword(
            win_data, "gyrotropic_smr_fixed_en_width", 0.0, dtype=float
        )
        d["gyrotropic_smr_max_arg"] = self._get_param_keyword(win_data, "gyrotropic_smr_max_arg", 5.0, dtype=float)
        d["gyrotropic_degen_thresh"] = self._get_param_keyword(win_data, "gyrotropic_degen_thresh", 0.0, dtype=float)
        d["gyrotropic_band_list"] = self._get_param_keyword(win_data, "gyrotropic_band_list", None, dtype=str)
        gyrotropic_box_center = self._get_param_keyword(win_data, "gyrotropic_box_center", "0.5 0.5 0.5", dtype=str)
        d["gyrotropic_box_center"] = [float(x) for x in gyrotropic_box_center.split()]
        gyrotropic_box_b1 = self._get_param_keyword(win_data, "gyrotropic_box_b1", "1.0 0.0 0.0", dtype=str)
        d["gyrotropic_box_b1"] = [float(x) for x in gyrotropic_box_b1.split()]
        gyrotropic_box_b2 = self._get_param_keyword(win_data, "gyrotropic_box_b2", "0.0 1.0 0.0", dtype=str)
        d["gyrotropic_box_b2"] = [float(x) for x in gyrotropic_box_b2.split()]
        gyrotropic_box_b3 = self._get_param_keyword(win_data, "gyrotropic_box_b3", "0.0 0.0 1.0", dtype=str)
        d["gyrotropic_box_b3"] = [float(x) for x in gyrotropic_box_b3.split()]

        d["boltzwann"] = self._get_param_keyword(win_data, "boltzwann", False, dtype=bool)

        num_fermi = 0
        found_fermi_energy = False
        fermi_energy_max = 0.0
        fermi_energy_min = 0.0
        fermi_energy_step = 0.0
        fermi_energy_list = []

        fermi_energy = self._get_param_keyword(win_data, "fermi_energy", 0.0, dtype=float)

        if fermi_energy is not None:
            # found_fermi_energy = True
            num_fermi = 1

        fermi_energy_scan = False
        fermi_energy_min = self._get_param_keyword(win_data, "fermi_energy_min", None, dtype=float)
        if fermi_energy_min is not None:
            if found_fermi_energy:
                raise Exception("Error: Cannot specify both fermi_energy and fermi_energy_min")

            fermi_energy_scan = True
            fermi_energy_max = fermi_energy_min + 1.0
            fermi_energy_max = self._get_param_keyword(win_data, "fermi_energy_max", None, dtype=float)

            if fermi_energy_max is not None and fermi_energy_max <= fermi_energy_min:
                raise Exception("Error: fermi_energy_max must be larger than fermi_energy_min")

            fermi_energy_step = 0.01
            fermi_energy_step = self._get_param_keyword(win_data, "fermi_energy_step", None, dtype=float)

            if fermi_energy_step is not None and fermi_energy_step <= 0.0:
                raise Exception("Error: fermi_energy_step must be positive")

            num_fermi = int(abs((fermi_energy_max - fermi_energy_min) / fermi_energy_step)) + 1

        if found_fermi_energy:
            fermi_energy_list = [fermi_energy]
        elif fermi_energy_scan:
            if num_fermi == 1:
                fermi_energy_step = 0.0
            else:
                fermi_energy_step = (fermi_energy_max - fermi_energy_min) / float(num_fermi - 1)

            fermi_energy_list = [fermi_energy_min + i * fermi_energy_step for i in range(num_fermi)]
        else:
            fermi_energy_list = [0.0]

        d["fermi_energy"] = fermi_energy
        d["fermi_energy_max"] = fermi_energy_max
        d["fermi_energy_min"] = fermi_energy_min
        d["fermi_energy_step"] = fermi_energy_step
        d["fermi_energy_list"] = fermi_energy_list
        d["num_fermi"] = num_fermi

        # shc
        d["shc_freq_scan"] = self._get_param_keyword(win_data, "shc_freq_scan", False, dtype=bool)
        d["shc_alpha"] = self._get_param_keyword(win_data, "shc_alpha", 1, dtype=int)
        d["shc_beta"] = self._get_param_keyword(win_data, "shc_beta", 2, dtype=int)
        d["shc_gamma"] = self._get_param_keyword(win_data, "shc_gamma", 3, dtype=int)
        d["shc_bandshift"] = self._get_param_keyword(win_data, "shc_bandshift", False, dtype=bool)
        d["shc_bandshift_firstband"] = self._get_param_keyword(win_data, "shc_bandshift_firstband", None, dtype=float)
        d["shc_bandshift_energyshift"] = self._get_param_keyword(
            win_data, "shc_bandshift_energyshift", 0.0, dtype=float
        )

        return d

    # ==================================================
    def _get_param_keyword(self, lines, keyword, default_value=None, dtype=int):
        data = None
        keys = []
        for line in lines:
            line = line.replace("\n", "")
            line = line.lstrip()
            if line.startswith(keyword):
                if len(line.split("=")) > 1:
                    key = line.split("=")[0]
                    key = key.replace(" ", "")
                    if key == keyword:
                        data = line.split("=")[1].split("!")[0]
                    assert key not in keys, key + " is defined more than once"
                    keys.append(key)
                elif len(line.split(":")) > 1:
                    key = line.split(":")[0]
                    key = key.replace(" ", "")
                    if key == keyword:
                        data = line.split(":")[1].split("!")[0]
                    assert key not in keys, key + " is defined more than once"
                    keys.append(key)

        if data is None:
            data = default_value
        else:
            if data.replace(" ", "").lower() in ("true", ".true."):
                data = True
            elif data.replace(" ", "").lower() in ("false", ".false."):
                data = False

        if data is None:
            return None

        return dtype(data)

    # ==================================================
    @property
    def gyrotropic_task_list(self):
        if self["gyrotropic_task"] is None:
            return []
        else:
            return [task.lower() for task in self["gyrotropic_task"].split("-")]

    # ==================================================
    @property
    def eval_K(self):
        return "k" in self.gyrotropic_task_list or "all" in self.gyrotropic_task_list

    # ==================================================
    @property
    def eval_C(self):
        return "c" in self.gyrotropic_task_list or "all" in self.gyrotropic_task_list

    # ==================================================
    @property
    def eval_D(self):
        return "d0" in self.gyrotropic_task_list or "all" in self.gyrotropic_task_list

    # ==================================================
    @property
    def eval_Dw(self):
        return "dw" in self.gyrotropic_task_list or "all" in self.gyrotropic_task_list

    # ==================================================
    @property
    def eval_spn(self):
        eval_spn = "spin" in self.gyrotropic_task_list or ("all" in self.gyrotropic_task_list and self["spinors"])
        if not (self.eval_K or self.eval_NOA):
            eval_spn = False
        return eval_spn

    # ==================================================
    @property
    def eval_NOA(self):
        return "noa" in self.gyrotropic_task_list or "all" in self.gyrotropic_task_list

    # ==================================================
    @property
    def eval_DOS(self):
        return "dos" in self.gyrotropic_task_list or "all" in self.gyrotropic_task_list

    # ==================================================
    @property
    def eval_DOS(self):
        return "dos" in self.gyrotropic_task_list or "all" in self.gyrotropic_task_list

    # ==================================================
    @property
    def gyrotropic_box(self):
        return np.array([self["gyrotropic_box_b1"], self["gyrotropic_box_b2"], self["gyrotropic_box_b3"]])

    # ==================================================
    @property
    def gyrotropic_box_center(self):
        return np.array(self["gyrotropic_box_center"])

    # ==================================================
    @property
    def gyrotropic_box_corner(self):
        return self.gyrotropic_box_center - 0.5 * (
            self.gyrotropic_box[0] + self.gyrotropic_box[1] + self.gyrotropic_box[2]
        )

    # ==================================================
    @classmethod
    def _default(cls):
        return _default
