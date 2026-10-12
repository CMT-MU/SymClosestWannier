"""
postcw (analyzer) on the graphene_pz example, which has no seedname.mmn: the tight-binding optical conductivity.
"""

import numpy as np

from symclosestwannier.analyzer.analyzer import analyzer
from symclosestwannier.cw.cw_creator import cw_creator
from symclosestwannier.cw.cw_info import _needs_mmn, _needs_uHu


# ==================================================
def test_postcw_computes_and_writes_the_response(make_case):
    """
    postcw computes the responses before writing them (calc_response was not called), and the tight-binding
    approximation needs no seedname.mmn (it was read for every berry task, and Band always built the position operator).
    """
    seedname = "graphene_pz"
    workdir = make_case(seedname, extra="write_info_data = true\nuse_tb_approximation = true\nverbose = false")
    with open(workdir / f"{seedname}.win", "a") as fp:
        fp.write("\nberry = true\nberry_task = kubo\nberry_kmesh = 6 6 1\nkubo_freq_max = 6.0\nkubo_freq_step = 0.5\n")
        fp.write("kubo_smr_fixed_en_width = 0.2\n")
    assert not (workdir / f"{seedname}.mmn").exists()

    cw_creator(seedname)
    analyzer(seedname)

    data = np.loadtxt(workdir / f"{seedname}-kubo_S_xx.dat")
    # frequencies 0, 0.5, ..., 6.0: kubo_freq_max is included, as in wannier90
    assert data.shape == (13, 3)
    np.testing.assert_allclose(data[:, 0], np.linspace(0.0, 6.0, 13))
    assert np.all(np.isfinite(data))
    assert np.abs(data[:, 2]).max() > 1.0


# ==================================================
def _flags(**kwargs):
    d = dict(
        calc_spreads=False,
        write_mmn=False,
        write_rmn=False,
        write_tb=False,
        tb_position=False,
        berry=False,
        berry_task="",
        use_tb_approximation=False,
        gyrotropic=False,
    )
    d.update(kwargs)
    return d


class _Win:
    def __init__(self, **kwargs):
        for k in ("eval_K", "eval_C", "eval_D", "eval_Dw", "eval_NOA"):
            setattr(self, k, kwargs.get(k, False))


def test_needs_mmn():
    assert not _needs_mmn(_flags())
    assert _needs_mmn(_flags(berry=True, berry_task="ahc"))
    assert not _needs_mmn(_flags(berry=True, berry_task="kubo", use_tb_approximation=True))
    assert _needs_mmn(_flags(write_tb=True))
    assert not _needs_mmn(_flags(write_tb=True, write_rmn=True, tb_position=True))
    assert _needs_mmn(_flags(gyrotropic=True))


def test_needs_uHu():
    assert not _needs_uHu(_flags(berry=True, berry_task="ahc"), _Win())
    assert _needs_uHu(_flags(berry=True, berry_task="morb"), _Win())
    assert not _needs_uHu(_flags(gyrotropic=True), _Win())
    assert _needs_uHu(_flags(gyrotropic=True), _Win(eval_K=True))
