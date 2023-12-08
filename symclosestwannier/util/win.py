"""
Win manages input file for wannier90.x, seedname.win file.
"""
import os
import gzip
import tarfile
import itertools

import numpy as np
import scipy.linalg


# ==================================================
class Win(dict):
    """
    Win manages input file for wannier90.x, seedname.win file.
    """

    # ==================================================
    def __init__(self, topdir, seedname, encoding="UTF-8"):
        """
        initialize the class.

        Args:
            topdir (str): directory of seedname.amn file.
            seedname (str): seedname.
            encoding (str, optional): encoding.
        """
        file_win = os.path.join(topdir, "{}.{}".format(seedname, "win"))

        self.update(self.read(file_win))

    # ==================================================
    def read(self, file_win):
        """
        read seedname.win file.

        Args:
            file_win (str): file name.

        Returns:
            dict:
                num_k (int): # of k points.
                num_bands (int): # of bands passed to the code.
                num_wann (int): # of CWFs.
                mp_grid (ndarray): dimensions of the Monkhorst-Pack grid of k-points.
                kpoints (ndarray): k-points used in DFT calculation, [[k1, k2, k3]] (crystal coordinate).
                kpoint (dict): representative k points.
                kpoint_path (str): k-points along high symmetry line in Brillouin zonen, [[k1, k2, k3]] (crystal coordinate).
                unit_cell_cart (ndarray): transform matrix, [a1,a2,a3].
                atoms_frac (ndarray): atomic positions in fractional coordinates with respect to the lattice vectors, {atom: [r1,r2,r3]}.
                atoms_cart (ndarray): atomic positions in cartesian coordinates, {atom: [rx,ry,rz]}.
        """
        d = {}

        if os.path.exists(file_win):
            with open(file_win) as fp:
                win_data = fp.readlines()
        else:
            raise Exception("failed to read win file: " + file_win)

        d["num_bands"] = self._get_param_keyword(win_data, "num_bands", 0, dtype=int)
        d["num_wann"] = self._get_param_keyword(win_data, "num_wann", 0, dtype=int)

        mp_grid = self._get_param_keyword(win_data, "mp_grid", [0, 0, 0], dtype=str)
        d["mp_grid"] = np.array([int(x) for x in mp_grid.split()])
        d["num_k"] = np.prod(d["mp_grid"])

        win_data = [v.replace("\n", "") for v in win_data]
        win_data_lower = [v.lower().replace("\n", "") for v in win_data]

        d |= {
            "kpoints": None,
            "kpoint": None,
            "kpoint_path": None,
            "unit_cell_cart": None,
            "atoms_frac": None,
            "atoms_cart": None,
        }

        for i, line in enumerate(win_data_lower):
            if "begin kpoints" in line:
                kpoints = np.genfromtxt(win_data[i + 1 : i + 1 + d["num_k"]], dtype=float)
                d["kpoints"] = np.mod(kpoints, 1)  # 0 <= kj < 1.0

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
                        kpoint[X] = np.array([float(ki1), float(ki2), float(ki3)])
                    if Y not in kpoint:
                        kpoint[Y] = np.array([float(kf1), float(kf2), float(kf3)])

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
                d["unit_cell_cart"] = unit_cell_cart

            if "begin atoms_frac" in line:
                ap_data = win_data[i + 1 : win_data_lower.index("end atoms_frac")]
                ap_data = [[vi for vi in v.split() if vi != ""] for v in ap_data]
                atoms_frac = {X: np.array([float(r1), float(r2), float(r3)]) for X, r1, r2, r3 in ap_data}
                d["atoms_frac"] = atoms_frac

            if "begin atoms_cart" in line:
                units = d["units"]
                n = 2 if units in ("bohr", "ang") else 1
                ap_data = win_data[i + n : win_data_lower.index("end atoms_cart")]
                ap_data = [[vi for vi in v.split() if vi != ""] for v in ap_data]
                atoms_cart = {X: np.array([float(r1), float(r2), float(r3)]) for X, r1, r2, r3 in ap_data}

                if units == "bohr":
                    atoms_cart = {X: v * 0.529177249 for X, v in atoms_cart.items()}

                d["atoms_cart"] = atoms_cart

        del d["units"]

        return d

    # ==================================================
    def _get_param_keyword(self, lines, keyword, default_value=None, dtype=int):
        data = None
        for line in lines:
            if line.startswith(keyword):
                assert data == None, keyword + " is defined more than once"
                if len(line.split("=")) > 1:
                    data = line.split("=")[1].split("!")[0]
                elif len(line.split(":")) > 1:
                    data = line.split(":")[1].split("!")[0]

                # data = line[2] if line[1] in ("=", ":") else line[1]

        if data == None:
            data = default_value

        if data == None:
            return None

        return dtype(data)

    # # ==================================================
    # def _get_param_keyword(self, lines, keyword, default_value=None, dtype=int):
    #     data = None
    #     for line in lines:
    #         if line.startswith(keyword):
    #             assert data == None, keyword + " is defined more than once"
    #             line = [vi for vi in line.replace("\n", "").split(" ") if vi != ""]
    #             if len(line) == 0:
    #                 continue

    #             if "!" in line[0]:
    #                 continue

    #             if "#" in line[0]:
    #                 continue

    #             data = line[2] if line[1] == "=" else line[1]

    #     if data == None:
    #         data = default_value

    #     if data == None:
    #         return None

    #     return dtype(data)
