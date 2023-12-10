"""
Win manages input file for wannier90.x, seedname.win file.
"""
import os
import numpy as np

from symclosestwannier.util.functions import kpoints_to_rpoints

_default_win = {
    "seedname": "cwannier",
    "num_k": 1,
    "num_bands": 1,
    "num_wann": 1,
    "mp_grid": [0, 0, 0],
    "kpoints": [[0, 0, 0]],
    "kpoint": None,
    "kpoint_path": None,
    "unit_cell_cart": None,
    "atoms_frac": None,
    "atoms_cart": None,
}


# ==================================================
class Win(dict):
    """
    Win manages input file for wannier90.x, seedname.win file.
    """

    # ==================================================
    def __init__(self, topdir=None, seedname=None, dic=None):
        """
        initialize the class.

        Args:
            topdir (str, optional): directory of seedname.win file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of Win.
        """
        super().__init__()

        if dic is None:
            file_name = os.path.join(topdir, "{}.{}".format(seedname, "win"))
            self.update(self.read(file_name))
            self["seedname"] = seedname
        else:
            self.update(dic)

    # ==================================================
    def read(self, file_name):
        """
        read seedname.win file.

        Args:
            file_name (str): file name.

        Returns:
            dict: dictionary form of seedname.win.
                - seedname       : seedname (str), ["cwannier"].
                - num_k          : # of k points (int), [1].
                - num_bands      : # of bands passed to the code (int), [1].
                - num_wann       : # of CWFs (int), [1].
                - mp_grid        : dimensions of the Monkhorst-Pack grid of k-points (list), [np.array([0, 0, 0])].
                - kpoints        : k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate) (list), [np.array([[0, 0, 0]])].
                - kpoint         : representative k points (dict), [None].
                - kpoint_path    : k-points along high symmetry line in Brillouin zonen, [[k1, k2, k3]] (crystal coordinate) (str), [None].
                - unit_cell_cart : transform matrix, [a1,a2,a3] (list), [None].
                - atoms_frac     : atomic positions in fractional coordinates with respect to the lattice vectors, {atom: [r1,r2,r3]} (dict), [None].
                - atoms_cart     : atomic positions in cartesian coordinates, {atom: [rx,ry,rz]} (dict), [None].
        """
        if os.path.exists(file_name):
            with open(file_name) as fp:
                win_data = fp.readlines()
        else:
            raise Exception("failed to read win file: " + file_name)

        d = Win._default_win().copy()

        win_data = [v.replace("\n", "") for v in win_data]
        win_data_lower = [v.lower().replace("\n", "") for v in win_data]

        d["num_bands"] = self._get_param_keyword(win_data, "num_bands", 0, dtype=int)
        d["num_wann"] = self._get_param_keyword(win_data, "num_wann", 0, dtype=int)
        mp_grid = self._get_param_keyword(win_data, "mp_grid", [0, 0, 0], dtype=str)
        d["mp_grid"] = [int(x) for x in mp_grid.split()]
        d["num_k"] = np.prod(d["mp_grid"])

        for i, line in enumerate(win_data_lower):
            if "begin kpoints" in line:
                kpoints = np.genfromtxt(win_data[i + 1 : i + 1 + d["num_k"]], dtype=float)
                d["kpoints"] = np.mod(kpoints, 1).tolist()  # 0 <= kj < 1.0

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
                d["atoms_frac"] = {k: (np.array(A).inv() @ np.array(v)).tolist() for k, v in d["atoms_cart"].items()}

        if d["atoms_frac"] is not None:
            if d["atoms_cart"] is None:
                A = d["unit_cell_cart"]
                d["atoms_cart"] = {k: (np.array(v) @ np.array(A)).tolist() for k, v in d["atoms_frac"].items()}

        d["rpoints"] = kpoints_to_rpoints(d["kpoints"]).tolist()

        return d

    # ==================================================
    def _get_param_keyword(self, lines, keyword, default_value=None, dtype=int):
        data = None
        for line in lines:
            line = line.replace("\n", "")
            if line.startswith(keyword):
                assert data == None, keyword + " is defined more than once"
                if len(line.split("=")) > 1:
                    data = line.split("=")[1].split("!")[0]
                elif len(line.split(":")) > 1:
                    data = line.split(":")[1].split("!")[0]

        if data == None:
            data = default_value

        if data == None:
            return None

        return dtype(data)

    # ==================================================
    @classmethod
    def _default_win(cls):
        return _default_win
