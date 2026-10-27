"""
tests for the K tensor of the kinetic magnetoelectric effect (gyrotropic_get_K), with the reference model of
test_tb_gauge.py.
"""

import contextlib
import io
import types

import numpy as np
import pytest

from symclosestwannier.analyzer import get_response as gr
from symclosestwannier.util.utility import fourier_transform_r_to_k

from test_tb_gauge import A, Model, make_cwi

N = 3


# ==================================================
class _CWInfo(dict):
    pass


def quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def gyro_cwi(model, efs, **kwargs):
    cwi = _CWInfo(
        {
            **make_cwi(model, False),
            "fermi_energy_list": efs,
            "num_fermi": len(efs),
            "unit_cell_volume": abs(np.linalg.det(A)),
            "transl_inv": False,
            "use_degen_pert": False,
            "degen_thr": 0.0,
            "gyrotropic_kmesh": [N, N, N],
            "gyrotropic_degen_thresh": 0.0,
            "gyrotropic_smr_max_arg": 5.0,
            "gyrotropic_smr_fixed_en_width": 0.1,
            "gyrotropic_smr_type": "gauss",
            "gyrotropic_band_list": None,
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


# ==================================================
def test_K_with_skipped_k_points():
    """
    k points far from the Fermi energy (gyrotropic_smr_max_arg, default 5) are skipped band by band; their
    contribution, below exp(-25), is negligible, so the K tensor equals the one without skipping.
    """
    model = Model()
    ops = model.operators_R()
    efs = [-0.45, 0.3]
    eta = 0.1

    kpoints = np.array([[i, j, k] for i in range(N) for j in range(N) for k in range(N)]) / N
    E = np.linalg.eigvalsh(fourier_transform_r_to_k(ops["HH_R"], kpoints, model.irvec, model.ndegen))
    arg = np.abs(E[None] - np.array(efs)[:, None, None]) / eta
    # for some band and Fermi energy, only part of the k points is kept (the case that failed)
    kept = (arg <= 5).sum(axis=1)  # (Fermi energy, band)
    assert np.any((kept > 0) & (kept < len(kpoints)))

    K_orb, K_spn = quiet(gr.gyrotropic_get_K, gyro_cwi(model, efs, gyrotropic_smr_fixed_en_width=eta), ops)
    ref_orb, ref_spn = quiet(
        gr.gyrotropic_get_K,
        gyro_cwi(model, efs, gyrotropic_smr_fixed_en_width=eta, gyrotropic_smr_max_arg=1e30),
        ops,
    )

    for K, ref in ((K_orb, ref_orb), (K_spn, ref_spn)):
        assert np.abs(ref).max() > 0
        np.testing.assert_allclose(K, ref, rtol=0, atol=1e-9 * np.abs(ref).max())


# ==================================================
@pytest.mark.parametrize("max_arg", [5.0, 1e30])
def test_K_with_all_k_points_skipped_for_a_band(max_arg):
    """
    a Fermi energy far from every band gives K = 0 (all skipped) or a negligible K (none skipped).
    """
    model = Model()
    ops = model.operators_R()

    K_orb, K_spn = quiet(gr.gyrotropic_get_K, gyro_cwi(model, [100.0], gyrotropic_smr_max_arg=max_arg), ops)

    assert np.abs(K_orb).max() < 1e-60 and np.abs(K_spn).max() < 1e-60
