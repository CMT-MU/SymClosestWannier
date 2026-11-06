"""
the closest Wannier gauge needs projections of full rank after the window function.

When fewer than num_wann states have appreciable weight in the energy window, the smallest singular values of the
window-weighted projections fall to the weight outside the window; the polar factor U(k) is then fixed by numerical
noise. pw2cw warns about it, and cwf_delta (the weight outside the window) can be raised to restore the rank.
"""

import numpy as np
import pytest

from symclosestwannier.cw.cw_creator import cw_creator
from symclosestwannier.cw.cw_model import CWModel, PROJECTION_RANK_TOL
from symclosestwannier.cw.cwin import CWin
from symclosestwannier.util.exceptions import SymCWInputError
from symclosestwannier.util.utility import projection_rank_ratio, weight_proj


# ==================================================
def projections(num_k=4, num_bands=12, num_wann=6, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(num_k, num_bands, num_wann)) + 1j * rng.normal(size=(num_k, num_bands, num_wann))
    E = np.sort(rng.uniform(-5.0, 15.0, size=(num_k, num_bands)), axis=1)
    return E, A


# ==================================================
def test_full_rank_projections():
    _, A = projections()
    assert projection_rank_ratio(A).min() > 1e-3


# ==================================================
def test_window_with_fewer_states_than_num_wann():
    # 6 Wannier functions, but only 4 states inside the window [-100, 0] at every k: the weights of the others are
    # of the size of the window tail, and the gauge becomes ill-defined; a finite cwf_delta restores the rank.
    E, A = projections()
    E[:, :4] = -1.0
    E[:, 4:] = np.linspace(5.0, 15.0, E.shape[1] - 4)
    w0 = weight_proj(E, -100.0, 0.0, 0.01, 0.1, 0.0)
    assert projection_rank_ratio(w0[:, :, None] * A).max() < PROJECTION_RANK_TOL

    w3 = weight_proj(E, -100.0, 0.0, 0.01, 0.1, 1e-3)
    assert projection_rank_ratio(w3[:, :, None] * A).min() > PROJECTION_RANK_TOL


# ==================================================
class _Model:
    def __init__(self, outfile):
        self._cwi = {"num_wann": 6}
        self._outfile = outfile


def test_warning_is_written(tmp_path, capsys):
    E, A = projections()
    E[:, :4] = -1.0
    E[:, 4:] = np.linspace(5.0, 15.0, E.shape[1] - 4)
    w = weight_proj(E, -100.0, 0.0, 0.01, 0.1, 0.0)
    out = tmp_path / "seed.cwout"
    ratio = CWModel._check_projection_rank(_Model(str(out)), w[:, :, None] * A)

    assert ratio.max() < PROJECTION_RANK_TOL
    assert "rank-deficient at 4 of 4 k points" in capsys.readouterr().err
    assert "increase cwf_delta" in out.read_text()


def test_no_warning_for_full_rank(tmp_path, capsys):
    _, A = projections()
    out = tmp_path / "seed.cwout"
    CWModel._check_projection_rank(_Model(str(out)), A)
    assert capsys.readouterr().err == ""
    assert not out.exists()


# ==================================================
def test_fewer_bands_than_wannier_functions():
    # with num_bands < num_wann the projections cannot have full column rank; SVD alone would not show it.
    A = np.ones((3, 1, 2), dtype=complex)
    assert np.all(projection_rank_ratio(A) == 0.0)


def test_non_finite_projections():
    _, A = projections()
    A[1, 0, 0] = np.nan
    ratio = projection_rank_ratio(A)
    assert ratio[1] == 0.0 and np.all(ratio[[0, 2, 3]] > 1e-3)


# ==================================================
EMPTY_WINDOW = "cwf_mu_max = -30.0\ncwf_mu_min = -40.0\ncwf_sigma_max = 0.01\ncwf_sigma_min = 0.01"


@pytest.mark.parametrize(
    "extra, warned",
    [
        ("", False),
        (EMPTY_WINDOW, True),
        (EMPTY_WINDOW + "\ncwf_delta = 0.5", False),
    ],
)
def test_pw2cw_warns_for_an_empty_window(make_case, capsys, extra, warned):
    # graphene p_z with an energy window that holds no band: all projections are suppressed unless cwf_delta > 0.
    workdir = make_case("graphene_pz", extra=extra)
    cw_creator("graphene_pz")
    assert ("rank-deficient" in capsys.readouterr().err) == warned
    assert ("rank-deficient" in (workdir / "graphene_pz.cwout").read_text()) == warned


# ==================================================
@pytest.mark.parametrize("value", ["0.0", "1e-12", "1e-3", "0.5", "0.999"])
def test_cwf_delta_accepted(tmp_path, value):
    (tmp_path / "seed.cwin").write_text(f"cwf_delta = {value}\n")
    assert CWin(str(tmp_path), "seed")["cwf_delta"] == float(value)


@pytest.mark.parametrize("value", ["1.0", "-1e-3", "nan", "inf"])
def test_cwf_delta_rejected(tmp_path, value):
    (tmp_path / "seed.cwin").write_text(f"cwf_delta = {value}\n")
    with pytest.raises(SymCWInputError):
        CWin(str(tmp_path), "seed")
