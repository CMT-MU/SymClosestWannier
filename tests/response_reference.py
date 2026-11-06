"""
reference values of the responses computed by postcw (berry_main, gyrotropic_main, spin_moment_main), for the
reference model of reference_model.py.

The values in tests/data/response_reference.npz were computed by the implementation of
symclosestwannier/analyzer/get_response.py before it is rewritten from the equations of the papers; test_response_reference.py
checks that any implementation reproduces them. Only the entry points are called, so that the internal structure of the
implementation is free. They check backward compatibility, not the physics. shc_fermi_adpt_kmesh was computed after the
correction of the adaptive k refinement of the SHC (Fermi energies below the threshold keep the point), so it differs
from wannier90 and from earlier versions.

usage: python tests/response_reference.py --write   (rewrites tests/data/response_reference.npz; only to add or change
reference cases with the implementation that defines them, never to make a new implementation pass)
"""

import contextlib
import io
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from symclosestwannier.analyzer import get_response as gr
from symclosestwannier.util.get_oper_R import get_v_R, to_tb_gauge

from reference_model import A, TAU, Model, make_cwi

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "response_reference.npz")
N = 3
# four Fermi energies, so that the Fermi-energy axis cannot be confused with a Cartesian one
EFS = [-0.45, 0.3, 1.2, 3.0]


# ==================================================
class _CWInfo(dict):
    pass


def base_cwi(model, tb_gauge, **kwargs):
    cwi = _CWInfo(
        {
            **make_cwi(model, tb_gauge),
            "unit_cell_volume": abs(np.linalg.det(A)),
            "B": 2 * np.pi * np.linalg.inv(A).T,  # reciprocal lattice vectors as rows
            "num_fermi": len(EFS),
            "fermi_energy_list": EFS,
            "fermi_energy": EFS[0],
            "spinors": True,
            "transl_inv": False,
            "use_tb_approximation": False,
            "use_degen_pert": False,
            "degen_thr": 0.0,
            "spin_decomp": False,
            "berry_kmesh": [N, N, N],
            "berry_curv_unit": "ang2",
            "berry_curv_adpt_kmesh": [1, 1, 1],
            "berry_curv_adpt_kmesh_thresh": 100.0,
            "kmesh": [N, N, N],
            # optical conductivity
            "kubo_freq_min": 0.0,
            "kubo_freq_max": 8.0,
            "kubo_freq_step": 0.5,
            "kubo_eigval_max": 1.0e4,
            "kubo_adpt_smr": False,
            "kubo_adpt_smr_fac": np.sqrt(2),
            "kubo_adpt_smr_max": 1.0,
            "kubo_smr_fixed_en_width": 0.2,
            "kubo_smr_type": "gauss",
            # shift current
            "sc_eta": 0.04,
            "sc_w_thr": 5.0,
            "sc_use_eta_corr": False,
            # spin Hall conductivity
            "shc_freq_scan": False,
            "shc_alpha": 1,
            "shc_beta": 2,
            "shc_gamma": 3,
            "shc_bandshift": False,
            "shc_bandshift_firstband": 0,
            "shc_bandshift_energyshift": 0.0,
            # gyrotropic
            "gyrotropic_kmesh": [N, N, N],
            "gyrotropic_degen_thresh": 0.0,
            "gyrotropic_smr_max_arg": 5.0,
            "gyrotropic_smr_fixed_en_width": 0.3,
            "gyrotropic_smr_type": "gauss",
            "gyrotropic_band_list": None,
            # spin moment
            "magnetic_field": 0.0,
            "magnetic_field_theta": 0.0,
            "magnetic_field_phi": 0.0,
            "g_factor": 2.0,
            **kwargs,
        }
    )
    cwi.win = types.SimpleNamespace(
        eval_K=True,
        eval_spn=True,
        eval_D=False,
        eval_Dw=False,
        eval_C=False,
        eval_NOA=False,
        gyrotropic_box=np.eye(3),
        gyrotropic_box_corner=np.zeros(3),
    )
    return cwi


def operators(model, cwi):
    ops = model.operators_R()
    if cwi["tb_gauge"]:
        ops = to_tb_gauge(ops, model.irvec, A, TAU)
    if cwi["use_tb_approximation"]:
        ops = {"HH_R": ops["HH_R"], "SS_R": ops["SS_R"], "v_R": get_v_R(cwi, ops["HH_R"])}
    return ops


# ==================================================
# (name, entry point, settings)
KUBO = dict(berry_task="kubo")
SHC = dict(berry_task="shc")
CASES = [
    ("ahc", "berry", dict(berry_task="ahc")),
    ("ahc_tb_gauge", "berry", dict(berry_task="ahc", tb_gauge=True)),
    # |Ω| exceeds 10 at some of the 27 points only (in Å² and in bohr² at different ones)
    ("ahc_adpt_kmesh", "berry", dict(berry_task="ahc", berry_curv_adpt_kmesh=[2, 2, 2], berry_curv_adpt_kmesh_thresh=10.0)),
    (
        "ahc_adpt_kmesh_bohr2",
        "berry",
        dict(berry_task="ahc", berry_curv_unit="bohr2", berry_curv_adpt_kmesh=[2, 2, 2], berry_curv_adpt_kmesh_thresh=10.0),
    ),
    ("morb", "berry", dict(berry_task="morb")),
    ("morb_tb_gauge", "berry", dict(berry_task="morb", tb_gauge=True)),
    ("kubo_gauss", "berry", dict(KUBO, num_fermi=1, fermi_energy_list=[0.3])),
    ("kubo_mp1", "berry", dict(KUBO, num_fermi=1, fermi_energy_list=[0.3], kubo_smr_type="m-p1")),
    ("kubo_cold", "berry", dict(KUBO, num_fermi=1, fermi_energy_list=[0.3], kubo_smr_type="m-v")),
    ("kubo_fd", "berry", dict(KUBO, num_fermi=1, fermi_energy_list=[0.3], kubo_smr_type="f-d")),
    ("kubo_adaptive", "berry", dict(KUBO, num_fermi=1, fermi_energy_list=[0.3], kubo_adpt_smr=True)),
    # the top band (2.1 to 6.4 eV) is cut at some k points
    ("kubo_eigval_max", "berry", dict(KUBO, num_fermi=1, fermi_energy_list=[0.3], kubo_eigval_max=5.0)),
    ("kubo_tb_gauge", "berry", dict(KUBO, num_fermi=1, fermi_energy_list=[0.3], tb_gauge=True)),
    ("kubo_tb_approximation", "berry", dict(KUBO, num_fermi=1, fermi_energy_list=[0.3], use_tb_approximation=True)),
    ("sc", "berry", dict(berry_task="sc", num_fermi=1, fermi_energy_list=[0.3])),
    ("sc_eta_corr", "berry", dict(berry_task="sc", num_fermi=1, fermi_energy_list=[0.3], sc_use_eta_corr=True)),
    ("sc_tb_gauge", "berry", dict(berry_task="sc", num_fermi=1, fermi_energy_list=[0.3], tb_gauge=True)),
    ("shc_fermi", "berry", dict(SHC)),
    ("shc_fermi_xzy", "berry", dict(SHC, shc_alpha=1, shc_beta=3, shc_gamma=2)),
    ("shc_fermi_adaptive", "berry", dict(SHC, kubo_adpt_smr=True)),
    # |Ω^z_xy| exceeds 10 Å² at some of the 27 points for the Fermi energies -0.45 and 3.0 eV, never for 0.3 and 1.2 eV
    ("shc_fermi_adpt_kmesh", "berry", dict(SHC, berry_curv_adpt_kmesh=[2, 2, 2], berry_curv_adpt_kmesh_thresh=10.0)),
    ("shc_freq", "berry", dict(SHC, shc_freq_scan=True, num_fermi=1, fermi_energy_list=[0.3])),
    (
        "shc_freq_bandshift",
        "berry",
        dict(
            SHC,
            shc_freq_scan=True,
            num_fermi=1,
            fermi_energy_list=[0.3],
            shc_bandshift=True,
            shc_bandshift_firstband=2,
            shc_bandshift_energyshift=0.5,
        ),
    ),
    ("gyro_K", "gyrotropic", dict()),
    ("gyro_K_mp1", "gyrotropic", dict(gyrotropic_smr_type="m-p1")),
    ("gyro_K_fd", "gyrotropic", dict(gyrotropic_smr_type="f-d")),
    ("spin_moment", "spin", dict(num_fermi=1, fermi_energy_list=[0.3])),
    (
        "spin_moment_zeeman",
        "spin",
        dict(num_fermi=1, fermi_energy_list=[0.3], zeeman_interaction=True, magnetic_field=50.0, magnetic_field_theta=30.0),
    ),
]

ENTRY = {"berry": "berry_main", "gyrotropic": "gyrotropic_main", "spin": "spin_moment_main"}


# ==================================================
def _flatten(prefix, value, out, none):
    if value is None:
        none.append(prefix)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}/{k}", v, out, none)
    elif isinstance(value, (tuple, list)) and any(isinstance(v, (dict, tuple, list, np.ndarray)) for v in value):
        for i, v in enumerate(value):
            _flatten(f"{prefix}/{i}", v, out, none)
    else:
        out[prefix] = np.asarray(value)


def compute(name, entry, settings, model=None):
    """
    the results of one case, as a flat dict {"name/key": array}; "name/None" lists the keys whose value is None (inactive
    results), so that the set of returned keys is part of the reference.
    """
    model = Model() if model is None else model
    settings = dict(settings)
    cwi = base_cwi(model, settings.pop("tb_gauge", False), **settings)
    ops = operators(model, cwi)
    with contextlib.redirect_stdout(io.StringIO()):
        res = getattr(gr, ENTRY[entry])(cwi, ops)
    out, none = {}, []
    _flatten(name, res, out, none)
    out[f"{name}/None"] = np.array(sorted(k.split("/", 1)[1] for k in none), dtype=str)
    return out


def compute_all():
    model = Model()
    out = {}
    for name, entry, settings in CASES:
        out.update(compute(name, entry, settings, model))
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: python tests/response_reference.py --write   (rewrites " + FILE + ")")
    data = compute_all()
    np.savez_compressed(FILE, **data)
    for k, v in data.items():
        if v.dtype.kind == "U":
            print(f"{k:45s} {', '.join(v)}")
        else:
            print(f"{k:45s} {str(v.shape):20s} max|.| = {np.max(np.abs(v)) if v.size else 0:.6g}")
