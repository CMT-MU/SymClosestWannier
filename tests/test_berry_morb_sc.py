"""
tests for the orbital magnetization (berry_task = morb) and the shift current (berry_task = sc),
with the reference model of test_tb_gauge.py.
"""

import contextlib
import io

import numpy as np
import pytest

from symclosestwannier.analyzer import get_response as gr
from symclosestwannier.util.get_oper_R import to_tb_gauge
from symclosestwannier.util.utility import fourier_transform_r_to_k

from test_tb_gauge import A, NUM_WANN, TAU, Model, make_cwi

N = 3
V = abs(np.linalg.det(A))
FREQ = dict(kubo_freq_min=0.5, kubo_freq_max=8.0, kubo_freq_step=0.5)


# ==================================================
def quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def operators(model, tb_gauge):
    ops = model.operators_R()
    return to_tb_gauge(ops, model.irvec, A, TAU, model.ndegen) if tb_gauge else ops


def morb_cwi(model, tb_gauge, efs):
    return {**make_cwi(model, tb_gauge), "fermi_energy_list": efs, "num_fermi": len(efs), "berry_kmesh": [N, N, N]}


def sc_cwi(model, tb_gauge, ef, sc_eta=0.04, eta_corr=False):
    return {
        **make_cwi(model, tb_gauge),
        **FREQ,
        "fermi_energy_list": [ef],
        "num_fermi": 1,
        "berry_kmesh": [N, N, N],
        "kubo_eigval_max": 100000,
        "kubo_adpt_smr": False,
        "kubo_smr_fixed_en_width": 0.2,
        "kubo_smr_type": "gauss",
        "sc_eta": sc_eta,
        "sc_w_thr": 5.0,
        "sc_use_eta_corr": eta_corr,
        "use_degen_pert": False,
        "degen_thr": 0.0,
        "unit_cell_volume": V,
    }


# ==================================================
def test_morb_is_the_average_of_the_k_terms():
    """
    M = -eV_au/bohr^2 <(-2Im g) + (-2Im h) - 2 E_F (-2Im f)>_k (wannier90 berry_main), and the local (LC) and itinerant
    (IC) parts add up to it.
    """
    model = Model()
    efs = [-0.45, 3.0]
    cwi = morb_cwi(model, False, efs)
    ops = operators(model, False)

    res = quiet(gr.berry_get_morb, cwi, ops)

    kpoints = gr._berry_kmesh(cwi)
    imf, img, imh = gr.berry_get_imfgh_klist(cwi, ops, kpoints, imf=True, img=True, imh=True)
    fac = -gr.eV_au / gr.bohr**2
    ef = np.array(efs)[:, None]
    expected = fac * (img.sum(axis=2) + imh.sum(axis=2) - 2 * ef[:, None] * imf.sum(axis=2)).mean(axis=1)

    np.testing.assert_allclose(res["morb"].sum(axis=1), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res["morb"], res["morb_LC"] + res["morb_IC"], rtol=0, atol=1e-14)
    assert np.abs(expected).max() > 0.01


# ==================================================
def test_morb_does_not_depend_on_tb_gauge():
    model = Model()
    efs = [-0.45, 3.0]
    M = [quiet(gr.berry_get_morb, morb_cwi(model, g, efs), operators(model, g))["morb"].sum(axis=1) for g in (False, True)]
    np.testing.assert_allclose(M[1], M[0], rtol=0, atol=1e-12)


# ==================================================
def test_shift_current_does_not_depend_on_tb_gauge():
    """
    the generalized derivative is gauge covariant for sc_eta -> 0; with a finite sc_eta the two gauges differ at O(sc_eta^2).
    """
    model = Model()
    sc = [quiet(gr.berry_get_sc, sc_cwi(model, g, 0.0, sc_eta=1e-4), operators(model, g))[1] for g in (False, True)]
    scale = np.abs(sc[0]).max()
    assert scale > 0
    np.testing.assert_allclose(sc[1], sc[0], rtol=0, atol=1e-7 * scale)


# ==================================================
@pytest.mark.parametrize("sc_eta", [0.04, 0.2])
def test_shift_current_eta_correction_removes_tb_gauge_dependence(sc_eta):
    """
    with a finite sc_eta the generalized derivative depends on the phase convention (tb_gauge); the correction of
    Eq. (19) of Lihm, PRB 103, 247101 (2021) (sc_use_eta_corr) makes the two conventions agree.
    """
    model = Model()
    sc = {
        corr: [
            quiet(gr.berry_get_sc, sc_cwi(model, g, 0.0, sc_eta=sc_eta, eta_corr=corr), operators(model, g))[1]
            for g in (False, True)
        ]
        for corr in (False, True)
    }
    scale = np.abs(sc[True][0]).max()

    assert np.abs(sc[False][1] - sc[False][0]).max() > 1e-6 * scale
    np.testing.assert_allclose(sc[True][1], sc[True][0], rtol=0, atol=1e-12 * scale)


# ==================================================
def test_shift_current_limits_and_invalid_widths():
    """
    sc_eta = 0 gives the limit of small sc_eta (not NaN), adaptive smearing gives finite values, and invalid widths are
    rejected.
    """
    model = Model()
    ops = operators(model, False)
    sc0 = quiet(gr.berry_get_sc, sc_cwi(model, False, 0.0, sc_eta=0.0, eta_corr=True), ops)[1]
    sc_small = quiet(gr.berry_get_sc, sc_cwi(model, False, 0.0, sc_eta=1e-6), ops)[1]
    assert np.all(np.isfinite(sc0))
    np.testing.assert_allclose(sc0, sc_small, rtol=0, atol=1e-8 * np.abs(sc_small).max())

    cwi = {**sc_cwi(model, False, 0.0), "kubo_adpt_smr": True, "kubo_adpt_smr_fac": np.sqrt(2), "kubo_adpt_smr_max": 1.0}
    cwi["B"] = 2 * np.pi * np.linalg.inv(A).T
    sc_adpt = quiet(gr.berry_get_sc, cwi, ops)[1]
    assert np.all(np.isfinite(sc_adpt)) and np.abs(sc_adpt).max() > 0

    with pytest.raises(gr.SymCWInputError, match="smearing width"):
        gr.berry_get_sc({**sc_cwi(model, False, 0.0), "kubo_smr_fixed_en_width": 0.0}, ops)
    with pytest.raises(gr.SymCWInputError, match="sc_eta"):
        gr.berry_get_sc(sc_cwi(model, False, 0.0, sc_eta=-0.1), ops)


# ==================================================
def test_shift_current_frequencies_and_fermi_energy():
    model = Model()
    freq, sc = quiet(gr.berry_get_sc, sc_cwi(model, False, 0.0), operators(model, False))

    np.testing.assert_allclose(freq, np.arange(0.5, 8.0 + 1e-9, 0.5))
    assert sc.shape == (3, 6, len(freq))

    # all bands occupied: no shift current.
    E = np.linalg.eigvalsh(
        fourier_transform_r_to_k(model.operators_R()["HH_R"], gr._berry_kmesh(sc_cwi(model, False, 0)), model.irvec, model.ndegen)
    )
    _, sc_full = quiet(gr.berry_get_sc, sc_cwi(model, False, E.max() + 1.0), operators(model, False))
    assert np.all(sc_full == 0)

    with pytest.raises(gr.SymCWInputError, match="zeeman"):
        gr.berry_get_sc({**sc_cwi(model, False, 0.0), "zeeman_interaction": True}, operators(model, False))

    cwi = sc_cwi(model, False, 0.0)
    cwi["num_fermi"] = 2
    with pytest.raises(gr.SymCWInputError, match="single Fermi energy"):
        gr.berry_get_sc(cwi, operators(model, False))


# ==================================================
@pytest.fixture
def wannierberri_system(tmp_path, monkeypatch):
    """
    wannier-berri system of the model: H and the position operator from a seedname_tb.dat written by symCW,
    BB and CC set in convention I with wannier-berri's Wannier centres.
    """
    import sys

    wb = pytest.importorskip("wannierberri")
    from wannierberri.system.system_tb import get_system_tb

    from symclosestwannier.cw.cw_manager import CWManager
    from symclosestwannier.cw.cw_model import CWModel

    monkeypatch.chdir(".")
    monkeypatch.setattr(sys, "path", list(sys.path))

    model = Model()
    ops = model.operators_R()
    cwi = {
        "seedname": "model",
        "num_wann": NUM_WANN,
        "unit_cell_cart": A.tolist(),
        "irvec": model.irvec.tolist(),
        "ndegen": model.ndegen.tolist(),
        "tb_gauge": False,
    }
    cwm = CWManager(topdir=str(tmp_path), verbose=False, parallel=False, formatter=False)
    CWModel(cwi, cwm, dic={}).write_tb(ops["HH_R"], ops["AA_R"], "model_tb.dat.cw")

    system = quiet(get_system_tb, tb_file=str(tmp_path / "model_tb.dat.cw"), berry=True, silent=True)
    assert np.array_equal(system.rvec.iRvec, model.irvec)

    tau = system.wannier_centers_cart @ np.linalg.inv(A)
    ops_I = to_tb_gauge(ops, model.irvec, A, tau, model.ndegen)
    system.set_R_mat("Ham", ops["HH_R"].copy(), reset=True)
    system.set_R_mat("AA", ops_I["AA_R"].transpose(1, 2, 3, 0).copy(), reset=True)
    system.set_R_mat("BB", ops_I["BB_R"].transpose(1, 2, 3, 0).copy())
    a, b = gr._alpha_A, gr._beta_A
    CC = np.array([1j * (ops_I["CC_R"][a[c], b[c]] - ops_I["CC_R"][b[c], a[c]]) for c in range(3)])
    system.set_R_mat("CC", CC.transpose(1, 2, 3, 0).copy())

    def run(calc):
        grid = wb.Grid(system, NKdiv=1, NKFFT=[N, N, N])
        r = quiet(
            wb.run,
            system,
            grid=grid,
            calculators={"x": calc},
            parallel=False,
            use_irred_kpt=False,
            symmetrize=False,
            fout_name=str(tmp_path / "wb"),
        )
        return r.results["x"].data

    return model, wb, run


# ==================================================
def test_morb_matches_wannierberri(wannierberri_system):
    model, wb, run = wannierberri_system
    efs = [-0.45, 3.0]
    M_wb = run(wb.calculators.static.Morb(Efermi=np.array(efs)))

    for g in (False, True):
        M = quiet(gr.berry_get_morb, morb_cwi(model, g, efs), operators(model, g))["morb"].sum(axis=1)
        np.testing.assert_allclose(M, M_wb, rtol=0, atol=1e-10)


# ==================================================
def test_shift_current_matches_wannierberri(wannierberri_system):
    """
    wannier-berri has no correction for a finite sc_eta, and works in its own gauge; with a small sc_eta both agree.
    """
    model, wb, run = wannierberri_system
    freq, sc = quiet(gr.berry_get_sc, sc_cwi(model, False, 0.0, sc_eta=1e-4), operators(model, False))
    calc = wb.calculators.dynamic.ShiftCurrent(
        sc_eta=1e-4, Efermi=[0.0], omega=freq, smr_fixed_width=0.2, smr_type="Gaussian", kBT=0
    )
    S_wb = run(calc)[0]  # (omega, a, b, c), A/V^2
    S_wb = np.array([[S_wb[:, a, b, c] for b, c in zip(gr._alpha_S, gr._beta_S)] for a in range(3)])

    assert np.abs(S_wb).max() > 0
    np.testing.assert_allclose(sc, S_wb, rtol=0, atol=1e-6 * np.abs(S_wb).max())


# ==================================================
@pytest.mark.parametrize("task", ["morb", "sc"])
def test_berry_main_and_output(tmp_path, monkeypatch, task):
    """
    berry_main computes morb and sc (they were left empty), and Response writes the results.
    """
    import sys

    from symclosestwannier.analyzer.response import Response
    from symclosestwannier.cw.cw_manager import CWManager

    monkeypatch.chdir(".")
    monkeypatch.setattr(sys, "path", list(sys.path))

    model = Model()
    cwi = morb_cwi(model, False, [0.0]) if task == "morb" else sc_cwi(model, False, 0.0)
    cwi.update(seedname="model", berry_task=task, transl_inv=False, use_tb_approximation=False)
    ops = operators(model, False)

    res = Response.__new__(Response)
    dict.__init__(res)
    res._cwi = cwi
    res._cwm = CWManager(topdir=str(tmp_path), verbose=False, parallel=False, formatter=False)
    res.update(quiet(gr.berry_main, cwi, ops))

    if task == "morb":
        res.write_morb()
        lines = [line for line in (tmp_path / "model-morb.dat").read_text().splitlines() if line.strip()]
        total = np.array(lines[-1].split()[-3:], dtype=float)
        np.testing.assert_allclose(total, res["morb"][0].sum(axis=0), rtol=0, atol=1e-12)
    else:
        res.write_sc()
        data = np.loadtxt(tmp_path / "model-sc_xyz.dat")
        np.testing.assert_allclose(data[:, 0], res["sc_freq"])
        np.testing.assert_allclose(data[:, 1], res["sc"][0, 5], rtol=1e-12, atol=0)  # x, (y, z)
        assert len(list(tmp_path.glob("model-sc_*.dat"))) == 18


# ==================================================
def test_more_processes_than_k_points(monkeypatch):
    """the k points were split with a step len(kpoints) // number of processes, which is zero for small meshes."""
    model = Model()
    cwi = {
        **morb_cwi(model, False, [-0.45, 3.0]),
        "berry_kmesh": [2, 2, 1],
        "berry_curv_unit": "ang2",
        "berry_curv_adpt_kmesh": [1, 1, 1],
        "berry_curv_adpt_kmesh_thresh": 1e30,
        "unit_cell_volume": V,
    }
    ops = operators(model, False)
    ahc = quiet(gr.berry_get_ahc, cwi, ops)
    monkeypatch.setattr(gr, "_num_proc", 9)
    np.testing.assert_allclose(quiet(gr.berry_get_ahc, cwi, ops), ahc, rtol=0, atol=1e-10)


def test_kubo_spin_decomposition_is_rejected():
    for f in (gr.berry_get_kubo, gr.berry_get_kubo_tb):
        with pytest.raises(gr.SymCWInputError, match="spin_decomp"):
            f({"spin_decomp": True}, {})


# ==================================================
@pytest.mark.parametrize("task, gyrotropic", [("ahc", False), ("kubo", True)])
def test_tb_approximation_only_for_kubo(tmp_path, monkeypatch, task, gyrotropic):
    import sys

    from test_tb_gauge import make_response

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(".")
    model = Model()
    res, _ = make_response(tmp_path, monkeypatch, model, False, task)
    res._cwi.update(use_tb_approximation=True, gyrotropic=gyrotropic)
    with pytest.raises(gr.SymCWInputError, match="use_tb_approximation"):
        res.set_operators()


def test_zeeman_interaction_builds_the_spin_operator(tmp_path, monkeypatch):
    import sys

    from test_tb_gauge import make_response

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(".")
    model = Model()
    res, ops = make_response(tmp_path, monkeypatch, model, False, "ahc")
    assert res["SS_R"] is None
    res._cwi.update(zeeman_interaction=True)
    res.set_operators()
    np.testing.assert_array_equal(res["SS_R"], ops["SS_R"])
