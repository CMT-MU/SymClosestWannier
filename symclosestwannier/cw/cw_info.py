# ****************************************************************** #
#                                                                    #
# This file is distributed as part of the symclosestwannier code and #
#     under the terms of the GNU General Public License. See the     #
#     file LICENSE in the root directory of the symclosestwannier    #
#      distribution, or http://www.gnu.org/licenses/gpl-3.0.txt      #
#                                                                    #
#          The symclosestwannier code is hosted on GitHub:           #
#                                                                    #
#            https://github.com/CMT-MU/SymClosestWannier             #
#                                                                    #
#                            written by                              #
#                        Rikuto Oiwa, RIKEN                          #
#                                                                    #
# ------------------------------------------------------------------ #
#                                                                    #
#                 cw_info: information for CWModel                   #
#                                                                    #
# ****************************************************************** #

import os
import numpy as np
from multipie import MaterialModel

from symclosestwannier.cw.cwin import CWin
from symclosestwannier.cw.win import Win
from symclosestwannier.cw.nnkp import Nnkp
from symclosestwannier.cw.eig import Eig
from symclosestwannier.cw.amn import Amn
from symclosestwannier.cw.mmn import Mmn, _sort_Mkb_nnkpts
from symclosestwannier.cw.umat import Umat
from symclosestwannier.cw.spn import Spn
from symclosestwannier.cw.uHu import UHu

from symclosestwannier.util.utility import (
    wigner_seitz,
    convert_w90_orbital,
    grid_path,
    surface_grid,
    site_offset,
    site_offsets,
    ket_samb_list,
)
from symclosestwannier.util.exceptions import SymCWInputError

_class_map = {
    "cwin": CWin,
    "win": Win,
    "nnkp": Nnkp,
    "eig": Eig,
    "amn": Amn,
    "mmn": Mmn,
    "umat": Umat,
    "spn": Spn,
    "uHu": UHu,
}


# ==================================================
def _needs_mmn(d):
    """
    seedname.mmn is needed for the position operator and the operators built from it (AA_R, BB_R, SR_R, ...).
    The velocity (write_vmn) and the tight-binding approximation of postcw (use_tb_approximation) use H only, and
    tb_position takes the position operator from the projection centres.
    """
    return bool(
        d["calc_spreads"]
        or d["write_mmn"]
        or (d["write_rmn"] and not d["tb_position"])
        or (d["write_tb"] and not d["tb_position"])
        or (d["berry"] and not d["use_tb_approximation"])
        or d["gyrotropic"]
    )


# ==================================================
def _needs_uHu(d, win):
    """
    seedname.uHu is needed for CC_R: the orbital magnetization (berry_task = morb) and the gyrotropic tasks.
    """
    if d["berry"] and d["berry_task"] == "morb":
        return True

    return bool(d["gyrotropic"] and np.any([win.eval_K, win.eval_C, win.eval_D, win.eval_Dw, win.eval_NOA]))


# ==================================================
class CWInfo(dict):
    """
    CWInfo manages information for CWModel, CWin, Win, Nnkp, Eig, Amn, Mmn, Umat, Spn, UHu.

    Attributes:
        _topdir (str): top directory.
        _seedname (str): seedname.
        _mm (MaterialModel): material model (MultiPie).
        _postcw (bool): postcw calculation?
    """

    # ==================================================
    def __init__(self, topdir=None, seedname="cwannier", dic=None, postcw=False):
        """
        CWInfo manages information for CWModel, CWin, Win, Nnkp, Eig, Amn, Mmn, Umat, Spn, UHu.

        Args:
            topdir (str, optional): directory of seedname.cwin file.
            seedname (str, optional): seedname.
            dic (dict, optional): dictionary of CWin.
            postcw (bool): postcw calculation?
        """
        super().__init__()

        self._topdir = topdir
        self._seedname = seedname
        self._postcw = postcw

        self._mm = None

        self.update(self.read(topdir, seedname, dic))

    # ==================================================
    def read(self, topdir, seedname, dic=None):
        """
        read seedname.cwin/win/nnkp/eig/amn/(mmn)/(umat)/(spn)/(uHu) files.

        Args:
            topdir (str): directory of seedname.cwin/win/eig/amn/mmn/nnkp files.
            seedname (str): seedname.
            dic (dict, optional): dictionary of CWin.

        Returns:
            dict: system information.
        """
        d = {}

        if dic is None:
            for C in _class_map.values():
                d.update(C._default())
        else:
            d = dic

        info_dict = {}
        for name, C in _class_map.items():
            if name in d:
                continue

            if name == "umat" and (d["restart"] != "w90"):
                continue
            if name == "mmn":
                if not _needs_mmn(d):
                    continue
            if name == "spn":
                if not np.any(
                    [
                        d["calc_spin_2d"],
                        d["write_spn"],
                        d["spin_decomp"],
                        d["spin_moment"],
                        d["zeeman_interaction"],
                        d["berry_task"] == "shc",
                        info_dict["win"].eval_spn,
                    ]
                ):
                    continue

            if name == "uHu":
                if not _needs_uHu(d, info_dict["win"]):
                    continue

                info = C(topdir, seedname, formatted=d["uHu_formatted"])
            elif name == "spn":
                info = C(topdir, seedname, formatted=d["spn_formatted"])
            else:
                info = C(topdir, seedname)

            for k, v in info.items():
                for name_, info_ in info_dict.items():
                    if k in info_:
                        v_ = info_[k]

                        if isinstance(v, list):
                            if not np.allclose(v, v_, 1e-6):
                                if k == "nnkpts":
                                    info = _sort_Mkb_nnkpts(info, info_dict["nnkp"])
                                    v = info[k]

                                    if np.allclose(v, v_, 1e-6):
                                        continue

                                msg = str(f"The values of {k} in {name} and {name_} files are inconsistent.")
                                raise SymCWInputError(msg)
                        else:
                            if v != v_:
                                msg = str(f"The values of {k} in {name} and {name_} files are inconsistent.")
                                raise SymCWInputError(msg)

            info_dict.update({name: info})
            d.update(info)

        if d["zeeman_interaction"]:
            if not d["spinors"]:
                raise SymCWInputError("zeeman_interaction = true requires spinor WFs (spinors = true in seedname.win).")

        #
        # additional information
        #
        d["A"] = np.array(d["unit_cell_cart"])
        d["B"] = 2 * np.pi * np.linalg.inv(d["A"]).T
        A, B = d["A"], d["B"]

        if self._postcw and dic is not None and "irvec" in dic and "ndegen" in dic:
            # a model read back from seedname.hdf5: keep the R vectors its operators are given on
            irvec, ndegen = np.array(dic["irvec"]), np.array(dic["ndegen"])
        else:
            irvec, ndegen = wigner_seitz(
                A, d["mp_grid"], ws_distance_tol=d["ws_distance_tol"], ws_search_size=d["ws_search_size"]
            )

        if d["kpoint"] is not None and d["kpoint_path"] is not None:
            kpoint = d["kpoint"]
            kpoint_path = d["kpoint_path"]
            N1 = d["N1"]
            B = np.linalg.inv(np.array(d["unit_cell_cart"], dtype=float))
            kpoints_path, k_linear, k_dis_pos = grid_path(kpoint, kpoint_path, N1, B)
            kpoints_path = kpoints_path.tolist()
            k_linear = k_linear.tolist()
        else:
            kpoints_path, k_linear, k_dis_pos = None, None, None

        d["unit_cell_volume"] = np.dot(A[0], np.cross(A[1], A[2]))
        d["irvec"] = irvec.tolist()
        d["ndegen"] = ndegen.tolist()
        d["kpoints_path"] = kpoints_path
        d["k_linear"] = k_linear
        d["k_dis_pos"] = k_dis_pos

        if d["fermi_surface"]:
            [[kmin_1, kmax_1, N1], [kmin_2, kmax_2, N2]] = d["fermi_surface_kmesh"]
            view = d["fermi_surface_view"]
            k3 = d["fermi_surface_const"]
            k1 = np.linspace(kmin_1, kmax_1, N1, endpoint=True)
            k2 = np.linspace(kmin_2, kmax_2, N2, endpoint=True)

            d["fermi_surface_grid_2d"], d["fermi_surface_grid"] = surface_grid(
                d["unit_cell_cart"], d["unit_cell_volume"], k1, k2, k3
            )

        if d["qpoint"] is not None and d["qpoint_path"] is not None:
            qpoint = d["qpoint"]
            qpoint_path = d["qpoint_path"]
            Nq1 = d["Nq1"]
            B = np.linalg.inv(np.array(d["unit_cell_cart"], dtype=float))
            qpoints_path, q_linear, q_dis_pos = grid_path(qpoint, qpoint_path, Nq1, B)
            qpoints_path = qpoints_path.tolist()
            q_linear = q_linear.tolist()
        else:
            qpoints_path, q_linear, q_dis_pos = None, None, None

        d["qpoints_path"] = qpoints_path
        d["q_linear"] = q_linear
        d["q_dis_pos"] = q_dis_pos

        if d["lindhard_surface"]:
            [[qmin_1, qmax_1, N1], [qmin_2, qmax_2, N2]] = d["lindhard_surface_qmesh"]
            view = d["lindhard_surface_view"]
            q3 = d["lindhard_surface_const"]
            q1 = np.linspace(qmin_1, qmax_1, N1, endpoint=True)
            q2 = np.linspace(qmin_2, qmax_2, N2, endpoint=True)

            d["qpoints_surface_grid_2d"], d["qpoints_surface_grid"] = surface_grid(
                d["unit_cell_cart"], d["unit_cell_volume"], q1, q2, q3
            )

        if d["symmetrization"]:
            self._mm = MaterialModel(topdir=os.path.abspath(topdir), verbose=True)
            self._mm.load(d["mp_seedname"])

            # ket
            molecule = self._mm.group.is_point_group
            site_dict = {
                (k, vi.sublattice): vi.position_primitive.tolist()
                for k, v in self._mm["site"]["cell"].items()
                for vi in v
                if vi.plus_set == 1
            }
            A = d["unit_cell_cart"]
            A_model = None if molecule else self._mm["unit_vector_primitive"]

            def _offset(pos, pos_model):
                # a molecule has no lattice: the positions must be the same.
                if molecule:
                    return [0, 0, 0] if np.allclose(pos, pos_model, rtol=1e-04, atol=1e-04) else None
                return site_offset(pos, pos_model, A, A_model)

            positions = [d["atom_pos_r"][pos_idx] for pos_idx in d["nw2n"]]

            auto = d["ket_amn"] == "auto"
            if auto:
                ket_amn = []
                for pos, l, m, r, s in zip(positions, d["nw2l"], d["nw2m"], d["nw2r"], d["nw2s"]):
                    match = [(name, sl) for (name, sl), pos_ in site_dict.items() if _offset(pos, pos_) is not None]
                    if len(match) == 0:
                        raise SymCWInputError(
                            f"the atomic position {pos} in seedname.nnkp is not found in the MultiPie model {d['mp_seedname']} "
                            "(also modulo the lattice vectors)."
                        )
                    name, sl = match[0]

                    orbital = convert_w90_orbital(l, m, r, s)
                    ket_amn.append([name, sl, l, orbital])

                d["ket_amn"] = ket_amn

            # lattice vector from the site of the MultiPie model to the projection centre in seedname.nnkp
            # (the operators are shifted by them before the SAMB decomposition, see CWModel._sym).
            ket = d["ket_amn"] if d["ket_amn"] is not None else ket_samb_list(self._mm)
            if len(ket) != d["num_wann"]:
                raise SymCWInputError(
                    f"the number of kets ({len(ket)}) in ket_amn (or the MultiPie model {d['mp_seedname']}) "
                    f"differs from num_wann = {d['num_wann']}."
                )
            for ket_i in ket:
                if (ket_i[0], ket_i[1]) not in site_dict:
                    raise SymCWInputError(
                        f"the site {ket_i[0]}_{ket_i[1]} of ket {ket_i} is not found in the MultiPie model {d['mp_seedname']}."
                    )

            if molecule:
                # a molecule has no lattice vectors (the positions of an explicit ket_amn are not checked).
                site_offset_list = [[0, 0, 0]] * len(ket)
            else:
                # the model may be translated as a whole from seedname.nnkp (only for an explicit ket_amn).
                pos_model = [site_dict[(ket_i[0], ket_i[1])] for ket_i in ket]
                site_offset_list, idx = site_offsets(positions, pos_model, A, A_model, translation=(not auto))
                if site_offset_list is None:
                    raise SymCWInputError(
                        f"the projection centre {positions[idx]} in seedname.nnkp of ket {ket[idx]} does not match the position "
                        f"{pos_model[idx]} of {ket[idx][0]}_{ket[idx][1]} in the MultiPie model {d['mp_seedname']} modulo the "
                        "lattice vectors (and a translation common to all the sites). "
                        "Check the order of ket_amn and the atomic positions."
                    )

            d["site_offset"] = site_offset_list

        return d

    # ==================================================
    def write(self, topdir, seedname):
        """
        read seedname.cwin/win/eig/amn/mmn/nnkp files.

        Args:
            topdir (str): directory of seedname.cwin/win/eig/amn/mmn/nnkp files.
            seedname (str): seedname.

        Returns:
            dict: system information.
        """
        pass

    # ==================================================
    @property
    def cwin(self):
        return CWin(dic={k: self[k] if k in self else v for k, v in CWin._default().items()})

    # ==================================================
    @property
    def win(self):
        return Win(dic={k: self[k] if k in self else v for k, v in Win._default().items()})

    # ==================================================
    @property
    def nnkp(self):
        return Nnkp(dic={k: self[k] if k in self else v for k, v in Nnkp._default().items()})

    # ==================================================
    @property
    def eig(self):
        return Eig(dic={k: self[k] if k in self else v for k, v in Eig._default().items()})

    # ==================================================
    @property
    def amn(self):
        return Amn(dic={k: self[k] if k in self else v for k, v in Amn._default().items()})

    # ==================================================
    @property
    def mmn(self):
        return Mmn(dic={k: self[k] if k in self else v for k, v in Mmn._default().items()})

    # ==================================================
    @property
    def umat(self):
        return Umat(dic={k: self[k] if k in self else v for k, v in Umat._default().items()})

    # ==================================================
    @property
    def spn(self):
        return Spn(dic={k: self[k] if k in self else v for k, v in Spn._default().items()})
