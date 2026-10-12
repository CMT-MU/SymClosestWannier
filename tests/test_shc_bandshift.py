"""
shc_bandshift: the bands shc_bandshift_firstband, ..., num_bands (1-based, as in wannier90) are shifted where the spin
Hall conductivity uses the energies.
"""

import numpy as np
import pytest

from symclosestwannier.analyzer import get_response
from symclosestwannier.util.get_oper_R import get_SHC_R, shc_bandshift

from test_get_oper_R import _CWInfo, _Nnkp


# ==================================================
@pytest.mark.parametrize("firstband, shifted", [(1, [1, 1, 1, 1]), (2, [0, 1, 1, 1]), (4, [0, 0, 0, 1]), (5, [0, 0, 0, 0])])
def test_shc_bandshift_bands(firstband, shifted):
    E = np.arange(8.0).reshape(2, 4)
    cwi = {"shc_bandshift": True, "shc_bandshift_firstband": firstband, "shc_bandshift_energyshift": -0.5}

    # bands firstband, ..., num_bands (1-based) are shifted, as in wannier90
    np.testing.assert_allclose(shc_bandshift(E, cwi) - E, [-0.5 * np.array(shifted)] * 2)
    np.testing.assert_allclose(E, np.arange(8.0).reshape(2, 4))
    assert shc_bandshift(E, {**cwi, "shc_bandshift": False, "shc_bandshift_firstband": None}) is E


# ==================================================
def make_shc_cwi(seed=0, **kwargs):
    """
    random Bloch data for get_SHC_R, num_bands = 4 > num_wann = 2.
    """
    rng = np.random.default_rng(seed)
    num_k, num_b, num_bands, num_wann = 2, 2, 4, 2

    def cplx(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    spn = cplx(3, num_k, num_bands, num_bands)
    cwi = _CWInfo(
        kpoints=np.array([[0, 0, 0], [0.5, 0, 0]]),
        irvec=np.array([[0, 0, 0], [1, 0, 0]]),
        num_k=num_k,
        wb=rng.uniform(0.5, 1.5, size=num_b),
        Ek=np.sort(rng.normal(size=(num_k, num_bands)), axis=1),
        Uk=cplx(num_k, num_bands, num_wann),
        pauli_spn=spn + spn.transpose(0, 1, 3, 2).conj(),
        Mkb=cplx(num_k, num_b, num_bands, num_bands),
        shc_bandshift=False,
        shc_bandshift_firstband=None,
        shc_bandshift_energyshift=0.0,
    )
    cwi.update(kwargs)
    cwi.nnkp = _Nnkp(rng.integers(0, num_k, size=(num_k, num_b)), rng.normal(size=(num_k, num_b, 3)))

    return cwi


# ==================================================
def test_get_SHC_R():
    """
    with the shift, SH_R and SHR_R are those of the Bloch energies shifted by hand; SR_R does not depend on the energies.
    """
    shift = {"shc_bandshift": True, "shc_bandshift_firstband": 3, "shc_bandshift_energyshift": 0.7}
    cwi = make_shc_cwi(**shift)
    Ek = cwi["Ek"].copy()
    ref = make_shc_cwi()
    ref["Ek"] = Ek + 0.7 * np.array([0, 0, 1, 1])
    plain = make_shc_cwi()

    SR, SHR, SH = get_SHC_R(cwi)
    SR_ref, SHR_ref, SH_ref = get_SHC_R(ref)
    SR_plain, SHR_plain, SH_plain = get_SHC_R(plain)

    np.testing.assert_allclose(SH, SH_ref, atol=1e-12)
    np.testing.assert_allclose(SHR, SHR_ref, atol=1e-12)
    np.testing.assert_allclose(SR, SR_plain, atol=1e-12)
    assert np.abs(SH - SH_plain).max() > 1e-3
    np.testing.assert_array_equal(cwi["Ek"], Ek)


# ==================================================
class _Stop(Exception):
    pass


# ==================================================
@pytest.mark.parametrize("firstband, shifted", [(1, [1, 1, 1]), (2, [0, 1, 1])])
def test_berry_get_shc_klist(monkeypatch, firstband, shifted):
    """
    the interpolated energies passed to the spin current are shifted from band shc_bandshift_firstband on.
    """
    rng = np.random.default_rng(1)
    num_wann = 3
    h1 = rng.normal(size=(num_wann, num_wann)) + 1j * rng.normal(size=(num_wann, num_wann))
    h0 = h1 + h1.T.conj()
    HH_R = np.array([h0, h1, h1.T.conj()])
    operators = {"HH_R": HH_R, "AA_R": np.zeros((3,) + HH_R.shape, dtype=complex)}
    cwi = {
        "tb_gauge": False,
        "unit_cell_cart": np.eye(3),
        "irvec": np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]),
        "ndegen": np.ones(3),
        "kubo_adpt_smr": False,
        "kubo_adpt_smr_fac": 1.0,
        "kubo_adpt_smr_max": 1.0,
        "kubo_smr_fixed_en_width": 0.0,
        "kubo_smr_type": "gauss",
        "kubo_eigval_max": 99999.0,
        "dis_froz_max": 100000.0,
        "kubo_freq_min": 0.0,
        "kubo_freq_max": 1.0,
        "kubo_freq_step": 0.5,
        "use_degen_pert": False,
        "degen_thr": 1e-4,
        "shc_alpha": 1,
        "shc_beta": 2,
        "shc_bandshift": False,
        "shc_bandshift_firstband": None,
        "shc_bandshift_energyshift": 0.0,
    }
    energies = []

    def capture(cwi, operators, kpoints, E, *args):
        energies.append(E)
        raise _Stop

    monkeypatch.setattr(get_response, "berry_get_js_k", capture)
    k = np.array([[0.13, 0.27, 0.41]])
    for c in (cwi, {**cwi, "shc_bandshift": True, "shc_bandshift_firstband": firstband, "shc_bandshift_energyshift": 0.3}):
        with pytest.raises(_Stop):
            get_response.berry_get_shc_klist(c, operators, k)

    np.testing.assert_allclose(energies[1] - energies[0], [0.3 * np.array(shifted)], atol=1e-12)
