"""
frequencies of the optical conductivity, the shift current and the ac spin Hall conductivity (kubo_freq_list of
wannier90), and the complex ac spin Hall conductivity, with the reference model of test_tb_gauge.py.
"""

import numpy as np
import pytest

from symclosestwannier.analyzer import get_response as gr
from symclosestwannier.util.exceptions import SymCWInputError

from test_tb_gauge import KPOINTS, Model, make_cwi


# ==================================================
@pytest.mark.parametrize(
    "fmin, fmax, fstep, expected",
    [
        (0.0, 6.0, 0.5, np.linspace(0.0, 6.0, 13)),  # kubo_freq_max is included
        (0.0, 1.0, 0.3, np.linspace(0.0, 1.0, 4)),  # nint(3.33) + 1 frequencies, evenly spaced up to kubo_freq_max
        (0.0, 1.0, 0.4, np.linspace(0.0, 1.0, 4)),  # nint(2.5) = 3: rounds half away from zero
        (1.0, 1.1, 1.0, np.array([1.0, 1.1])),  # at least two frequencies
        # wannier90 does not check kubo_freq_max > kubo_freq_min: still two frequencies, from min to max
        (1.0, 1.0, 0.5, np.array([1.0, 1.0])),
        (1.0, 0.0, 0.3, np.array([1.0, 0.0])),
    ],
)
def test_kubo_frequencies(fmin, fmax, fstep, expected):
    freq = gr.kubo_frequencies({"kubo_freq_min": fmin, "kubo_freq_max": fmax, "kubo_freq_step": fstep})
    np.testing.assert_allclose(freq, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("fstep", [0.0, -0.5, np.nan, np.inf])
def test_kubo_frequencies_invalid_step(fstep):
    with pytest.raises(SymCWInputError, match="kubo_freq_step must be positive"):
        gr.kubo_frequencies({"kubo_freq_min": 0.0, "kubo_freq_max": 1.0, "kubo_freq_step": fstep})


# ==================================================
def shc_cwi(model, freq_scan):
    return {
        **make_cwi(model, False),
        "shc_beta": 2,
        "berry_kmesh": [3, 3, 3],
        "fermi_energy": 0.0,
        "shc_freq_scan": freq_scan,
        "kubo_adpt_smr": False,
        "kubo_adpt_smr_fac": np.sqrt(2),
        "kubo_adpt_smr_max": 1.0,
        "kubo_smr_fixed_en_width": 0.2,
        "kubo_smr_type": "gauss",
        "kubo_eigval_max": 99999.0,
        "dis_froz_max": 100000.0,
        "kubo_freq_min": 0.0,
        "kubo_freq_max": 4.0,
        "kubo_freq_step": 0.5,
        "use_degen_pert": False,
        "degen_thr": 0.0,
        "shc_bandshift": False,
        "shc_bandshift_firstband": None,
        "shc_bandshift_energyshift": 0.0,
    }


def test_shc_frequency_scan_is_complex():
    """
    the ac spin Hall conductivity keeps its imaginary part; at zero frequency it is real and equal to the dc one.
    """
    model = Model()
    ops = model.operators_R()

    shc_freq = gr.berry_get_shc_klist(shc_cwi(model, True), ops, KPOINTS)  # (frequency, k)
    shc_fermi = gr.berry_get_shc_klist(shc_cwi(model, False), ops, KPOINTS)  # (Fermi energy, k)

    assert shc_freq.shape == (9, len(KPOINTS))
    scale = np.abs(shc_freq).max()
    assert np.abs(shc_freq.imag[1:]).max() > 1e-3 * scale
    np.testing.assert_allclose(shc_freq[0], shc_fermi[0], rtol=0, atol=1e-12 * scale)
