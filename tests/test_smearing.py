"""
smearing settings of the optical conductivity and the gyrotropic K tensor (kubo_smr_type, gyrotropic_smr_type,
fixed widths), with the reference model of test_tb_gauge.py.
"""

import contextlib
import io
import types

import numpy as np
import pytest

from symclosestwannier.analyzer import get_response as gr
from symclosestwannier.cw.win import Win
from symclosestwannier.util.exceptions import SymCWInputError
from symclosestwannier.util.get_oper_R import get_v_R

from test_berry_morb_sc import sc_cwi
from test_tb_gauge import A, Model, make_cwi

N = 3
SMR_TYPES = ("gauss", "m-p0", "m-p1")


# ==================================================
class _CWInfo(dict):
    pass


def quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def kubo_cwi(model, **kwargs):
    return {
        **make_cwi(model, False),
        "fermi_energy": 0.3,
        "fermi_energy_list": [0.3],
        "num_fermi": 1,
        "berry_kmesh": [N, N, N],
        "unit_cell_volume": abs(np.linalg.det(A)),
        "spin_decomp": False,
        "kubo_adpt_smr": False,
        "kubo_adpt_smr_fac": np.sqrt(2),
        "kubo_adpt_smr_max": 1.0,
        "kubo_smr_fixed_en_width": 0.2,
        "kubo_smr_type": "gauss",
        "kubo_eigval_max": 1e4,
        "kubo_freq_min": 0.5,
        "kubo_freq_max": 8.0,
        "kubo_freq_step": 0.5,
        "use_degen_pert": False,
        "degen_thr": 0.0,
        **kwargs,
    }


def kubo_ops(model, cwi):
    """operators of the model, with the velocity of the tight-binding approximation (berry_get_kubo_tb)."""
    ops = model.operators_R()
    return {**ops, "v_R": get_v_R(cwi, ops["HH_R"])}


def gyro_cwi(model, **kwargs):
    cwi = _CWInfo(
        {
            **make_cwi(model, False),
            "fermi_energy_list": [0.3],
            "num_fermi": 1,
            "unit_cell_volume": abs(np.linalg.det(A)),
            "transl_inv": False,
            "use_degen_pert": False,
            "degen_thr": 0.0,
            "gyrotropic_kmesh": [N, N, N],
            "gyrotropic_degen_thresh": -1.0,
            "gyrotropic_smr_max_arg": 1e30,
            "gyrotropic_smr_fixed_en_width": 0.3,
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
@pytest.mark.parametrize(
    "text, kubo, gyrotropic",
    [
        ("", "gauss", "gauss"),
        ("smr_type = m-v\n", "m-v", "m-v"),
        ("smr_type = m-v\nkubo_smr_type = m-p2\ngyrotropic_smr_type = f-d\n", "m-p2", "f-d"),
        ("smr_type = m-v\nkubo_smr_type = m-p2\n", "m-p2", "m-v"),
        ("smr_type = m-v\ngyrotropic_smr_type = f-d\n", "m-v", "f-d"),
    ],
)
def test_smearing_type_keywords(tmp_path, text, kubo, gyrotropic):
    """kubo_smr_type and gyrotropic_smr_type override smr_type, as in wannier90."""
    cell = "begin unit_cell_cart\nang\n1 0 0\n0 1 0\n0 0 1\nend unit_cell_cart\nbegin atoms_frac\nA 0 0 0\nend atoms_frac\n"
    (tmp_path / "seed.win").write_text("num_bands = 2\nnum_wann = 2\n" + text + cell)

    win = Win(str(tmp_path), "seed")

    assert win["kubo_smr_type"] == kubo and win["gyrotropic_smr_type"] == gyrotropic


# ==================================================
@pytest.mark.parametrize(
    "smr_type, index", [("gauss", 0), ("m-p", 1), ("m-p0", 0), ("m-p2", 2), ("m-v", -1), ("cold", -1), ("f-d", -99)]
)
def test_smearing_index(smr_type, index):
    assert gr._smearing_index(smr_type) == index


@pytest.mark.parametrize("smr_type", ["lorentz", "m-pfoo", "m-p-1", "m-p-99", "m-p1.5", "m-p²"])
def test_smearing_index_invalid(smr_type):
    with pytest.raises(SymCWInputError, match="unknown smearing type"):
        gr._smearing_index(smr_type)


# ==================================================
def assert_methfessel_paxton(results):
    """m-p0 is the Gaussian, m-p1 differs from it."""
    assert all(np.all(np.isfinite(r)) for r in results.values())
    scale = np.abs(results["gauss"]).max()
    assert scale > 0
    np.testing.assert_allclose(results["m-p0"], results["gauss"], rtol=0, atol=1e-12 * scale)
    assert np.abs(results["m-p1"] - results["gauss"]).max() > 1e-3 * scale


@pytest.mark.parametrize("f", [gr.berry_get_kubo, gr.berry_get_kubo_tb])
def test_methfessel_paxton_kubo(f):
    model = Model()
    ops = kubo_ops(model, kubo_cwi(model))
    assert_methfessel_paxton({t: quiet(f, kubo_cwi(model, kubo_smr_type=t), ops)[0] for t in SMR_TYPES})


def test_methfessel_paxton_gyrotropic():
    model = Model()
    ops = model.operators_R()
    gyro = {t: quiet(gr.gyrotropic_get_K, gyro_cwi(model, gyrotropic_smr_type=t), ops) for t in SMR_TYPES}
    for part in range(2):
        assert_methfessel_paxton({t: K[part] for t, K in gyro.items()})


def test_shift_current_smearing_type():
    """the shift current follows kubo_smr_type."""
    model = Model()
    ops = model.operators_R()
    assert_methfessel_paxton(
        {t: quiet(gr.berry_get_sc, {**sc_cwi(model, False, 0.0), "kubo_smr_type": t}, ops)[1] for t in SMR_TYPES}
    )
    with pytest.raises(SymCWInputError, match="unknown smearing type"):
        gr.berry_get_sc({**sc_cwi(model, False, 0.0), "kubo_smr_type": "lorentz"}, ops)


# ==================================================
@pytest.mark.parametrize("f", [gr.berry_get_kubo, gr.berry_get_kubo_tb])
def test_kubo_invalid_smearing(f):
    model = Model()
    ops = model.operators_R()
    with pytest.raises(SymCWInputError, match="unknown smearing type"):
        f(kubo_cwi(model, kubo_smr_type="lorentz"), ops)
    for width in (0.0, -0.1):
        with pytest.raises(SymCWInputError, match="positive smearing width"):
            f(kubo_cwi(model, kubo_smr_fixed_en_width=width), ops)


def test_gyrotropic_invalid_smearing():
    model = Model()
    ops = model.operators_R()
    with pytest.raises(SymCWInputError, match="unknown smearing type"):
        gr.gyrotropic_get_K(gyro_cwi(model, gyrotropic_smr_type="lorentz"), ops)
    for width in (0.0, -0.1):
        with pytest.raises(SymCWInputError, match="positive smearing width"):
            gr.gyrotropic_get_K(gyro_cwi(model, gyrotropic_smr_fixed_en_width=width), ops)


@pytest.mark.parametrize("f", [gr.berry_get_kubo, gr.berry_get_kubo_tb])
def test_kubo_adaptive_smearing_allows_zero_fixed_width(f):
    model = Model()
    ops = kubo_ops(model, kubo_cwi(model))
    B = 2 * np.pi * np.linalg.inv(A).T  # reciprocal lattice, for the adaptive width
    H = quiet(f, kubo_cwi(model, kubo_adpt_smr=True, kubo_smr_fixed_en_width=0.0, B=B), ops)[0]
    assert np.all(np.isfinite(H)) and np.abs(H).max() > 0
