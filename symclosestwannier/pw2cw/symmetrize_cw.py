"""
Symmetrized Closest Wannier (CW) tight-binding (TB) model by using Symmetry-Adapted Multipole Basis (SAMB).
"""
import sympy as sp
import numpy as np

from gcoreutils.nsarray import NSArray
from multipie.tag.tag_multipole import TagMultipole

from symclosestwannier.pw2cw.cw import CW


# ==================================================
class SymCW(dict):
    """
    symmetrized Closest Wannier (CW) tight-binding (TB) model.

    Attributes:
        _cw (CW): Closest Wannier (CW) tight-binding (TB) model.
    """

    # ==================================================
    def __init__(self, cw):
        """
        initialize the class.

        Args:
            cw (CW): Closest Wannier (CW) tight-binding (TB) model.
        """
        self._cw = cw

        Hk = self._cw["Hk"]
        Hr_dict = CW.matrix_dict_r(self._cw["Hr"], self._cw["rpoints"])
        Sr_dict = CW.matrix_dict_r(self._cw["Sr"], self._cw["rpoints"])
        Hr_nonortho_dict = CW.matrix_dict_r(self._cw["Hr_nonortho"], self._cw["rpoints"])

        #####

        msg = "   - reading output of multipie ... "
        self._cw._cwm.log(msg, None, end="", file=self._cw._outfile, mode="a")
        self._cw._cwm.set_stamp()

        model = self._cw.read(f"{self._cwi['mp_seedname']}_model.py", dir=self._cwi["mp_outdir"])
        samb = self.read(f"{self._cwi['mp_seedname']}_samb.py", dir=self._cwi["mp_outdir"])

        try:
            mat = self.read(f"{self._cwi['mp_seedname']}_matrix.pkl", dir=self._cwi["mp_outdir"])
        except:
            mat = self.read(f"{self._cwi['mp_seedname']}_matrix.py", dir=self._cwi["mp_outdir"])

        ket_samb = model["info"]["ket"]

        # sort orbitals
        ket_amn = self._cw["ket_amn"]
        if ket_amn is not None:
            idx_list = [ket_amn.index(o) for o in ket_samb]
            Hk = Hk[:, idx_list, :]
            Hk = Hk[:, :, idx_list]

            idx_list = [ket_samb.index(o) for o in ket_amn]
            Hr_dict = {(n1, n2, n3, idx_list[a], idx_list[b]): v for (n1, n2, n3, a, b), v in Hr_dict.items()}
            Sr_dict = {(n1, n2, n3, idx_list[a], idx_list[b]): v for (n1, n2, n3, a, b), v in Sr_dict.items()}

            Hr_nonortho_dict = {
                (n1, n2, n3, idx_list[a], idx_list[b]): v for (n1, n2, n3, a, b), v in Hr_nonortho_dict.items()
            }

        if irreps == "all":
            irreps = model["info"]["generate"]["irrep"]
        elif irreps == "full":
            irreps = [model["info"]["generate"]["irrep"][0]]

        for zj, (tag, _) in samb["data"]["Z"].items():
            if TagMultipole(tag).irrep not in irreps:
                del mat["matrix"][zj]

        tag_dict = {zj: tag for zj, (tag, _) in samb["data"]["Z"].items()}
        Zr_dict = {
            (zj, tag_dict[zj]): {tuple(sp.sympify(k)): complex(sp.sympify(v)) for k, v in d.items()}
            for zj, d in mat["matrix"].items()
        }
        mat["matrix"] = {
            zj: {tuple(sp.sympify(k)): complex(sp.sympify(v)) for k, v in d.items()} for zj, d in mat["matrix"].items()
        }

        lattice = model["info"]["group"][1].split("/")[1].replace(" ", "")[0]
        if lattice != "P":
            cell_site = {}
            for site, v in mat["cell_site"].items():
                if "(" in site and ")" in site:
                    if "(1)" in site:
                        cell_site[site[:-3]] = v
                else:
                    cell_site[site] = v

            mat["cell_site"] = cell_site

        self._cwm.log("done", file=self._outfile, mode="a")

        #####

        self._print(
            "   - decomposing Hamiltonian Hr as linear combination of SAMBs ... ",
            end="",
            mode="a",
        )
        start = time.time()

        z = CW.samb_decomp(Hr_dict, Zr_dict)

        end = time.time()
        self._print(f"done ({'{:.2f}'.format(end - start)} [sec])", mode="a")

        #####

        self._print("   - decomposing overlap Sr as linear combination of SAMBs ... ", end="", mode="a")
        start = time.time()

        s = CW.samb_decomp(Sr_dict, Zr_dict)

        end = time.time()
        self._print(f"done ({'{:.2f}'.format(end - start)} [sec])", mode="a")

        #####

        self._print(
            "   - decomposing non-orthogonal Hamiltonian Hr as linear combination of SAMBs ... ",
            end="",
            mode="a",
        )
        start = time.time()

        z_nonortho = CW.samb_decomp(Hr_nonortho_dict, Zr_dict)

        end = time.time()
        self._print(f"done ({'{:.2f}'.format(end - start)} [sec])", mode="a")

        #####

        rpoints_mp = [(n1, n2, n3) for Zj_dict in Zr_dict.values() for (n1, n2, n3, _, _) in Zj_dict.keys()]
        rpoints_mp = sorted(list(set(rpoints_mp)), key=rpoints_mp.index)

        if not mat["molecule"]:
            kpoint = {i: NSArray(j, "vector", fmt="value") for i, j in self["info"]["kpoint"].items()}
            kpoint_path = self["info"]["kpoint_path"]
            N1 = self["info"]["N1"]
            A = model["detail"]["A"]
            B = NSArray(A, "matrix", fmt="value").T.inverse()
            kpoints_path, _, _ = NSArray.grid_path(kpoint, kpoint_path, N1, B)

        #####

        self["info"]["self._cwi['mp_outdir']"] = self._cwi["mp_outdir"]
        self["info"]["mp_seedname"] = mp_seedname
        self["info"]["ket_amn"] = ket_amn
        self["info"]["irreps"] = irreps

        self["data"] = {
            "z": z,
            "s": s,
            "z_nonortho": z_nonortho,
            #
            "rpoints_mp": rpoints_mp,
        } | self["data"]

        self["data"]["matrix_dict"] = mat

        #####

        self._print("   - evaluating fitting accuracy ... ", end="\n", mode="a")
        start = time.time()

        Ek_grid, _ = np.linalg.eigh(Hk)
        Ek_grid_sym, _ = np.linalg.eigh(self.Hk_sym)
        num_k, num_wann = Ek_grid_sym.shape
        Ek_RMSE_grid = np.sum(np.abs(Ek_grid_sym - Ek_grid)) / num_k / num_wann * 1000  # [meV]

        self._print(
            f"     * RMSE of eigen values between CW and Symmetry-Adapted CW models (grid) = {'{:.4f}'.format(Ek_RMSE_grid)} [meV]",
            end="\n",
            mode="a",
        )

        #####

        if not mat["molecule"]:
            Hk_path = CW.interpolate(Hk, kpoints, kpoints_path, rpoints)
            Ek_path, _ = np.linalg.eigh(Hk_path)
            Ek_path_sym, _ = np.linalg.eigh(self.Hk_sym_path)
            num_k, num_wann = Ek_path_sym.shape
            Ek_RMSE_path = np.sum(np.abs(Ek_path_sym - Ek_path)) / num_k / num_wann * 1000  # [meV]
            self._print(
                f"     * RMSE of eigen values between CW and Symmetry-Adapted CW models (path) = {'{:.4f}'.format(Ek_RMSE_path)} [meV]",
                end="\n",
                mode="a",
            )

        end = time.time()
        self._print(f"    done ({'{:.2f}'.format(end - start)} [sec])", mode="a")

        #####

        if Ek_RMSE_grid is not None:
            self["data"] = {"Ek_RMSE_grid": Ek_RMSE_grid} | self["data"]

        if not mat["molecule"]:
            self["data"] = {"Ek_RMSE_path": Ek_RMSE_path} | self["data"]

        self["info"]["symmetrization"] = True

        #####

        end0 = time.time()
        self._print(f"  done ({'{:.2f}'.format(end0 - start0)} [sec])", mode="a")

        return s, z, z_nonortho, rpoints_mp, matrix_dict
