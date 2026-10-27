"""
Fermi energies of seedname.win (fermi_energy, or a scan with fermi_energy_min/max/step), as in wannier90.
"""

import numpy as np
import pytest

from symclosestwannier.util.exceptions import SymCWInputError

from test_tb_gauge import A, Model, make_cwi
from test_win import write_win


# ==================================================
def test_no_fermi_energy(tmp_path):
    win = write_win(tmp_path, "")
    assert win["fermi_energy"] == 0.0 and win["fermi_energy_list"] == [0.0] and win["num_fermi"] == 1


def test_single_fermi_energy(tmp_path):
    win = write_win(tmp_path, "fermi_energy = -2.5\n")
    assert win["fermi_energy"] == -2.5 and win["fermi_energy_list"] == [-2.5] and win["num_fermi"] == 1


# ==================================================
@pytest.mark.parametrize(
    "text, efs",
    [
        ("fermi_energy_min = -3.0\nfermi_energy_max = -1.0\nfermi_energy_step = 0.5\n", np.linspace(-3.0, -1.0, 5)),
        # (max - min) / step = 6.67: nint gives 8 energies, evenly spaced between min and max
        ("fermi_energy_min = -3.0\nfermi_energy_max = -1.0\nfermi_energy_step = 0.3\n", np.linspace(-3.0, -1.0, 8)),
        # (max - min) / step = 2.5: nint rounds half away from zero (4 energies)
        ("fermi_energy_min = 0.0\nfermi_energy_max = 1.0\nfermi_energy_step = 0.4\n", np.linspace(0.0, 1.0, 4)),
        # defaults: fermi_energy_max = fermi_energy_min + 1, fermi_energy_step = 0.01
        ("fermi_energy_min = 2.0\n", np.linspace(2.0, 3.0, 101)),
        ("fermi_energy_min = 2.0\nfermi_energy_max = 2.5\n", np.linspace(2.0, 2.5, 51)),
        ("fermi_energy_min = 2.0\nfermi_energy_step = 0.25\n", np.linspace(2.0, 3.0, 5)),
    ],
)
def test_fermi_energy_scan(tmp_path, text, efs):
    win = write_win(tmp_path, text)
    assert win["num_fermi"] == len(efs)
    np.testing.assert_allclose(win["fermi_energy_list"], efs, rtol=0, atol=1e-12)


# ==================================================
@pytest.mark.parametrize(
    "text, match",
    [
        ("fermi_energy = 0.0\nfermi_energy_min = -1.0\n", "both fermi_energy and fermi_energy_min"),
        ("fermi_energy_min = 1.0\nfermi_energy_max = 1.0\n", "fermi_energy_max must be larger"),
        ("fermi_energy_min = 1.0\nfermi_energy_step = 0.0\n", "fermi_energy_step must be positive"),
    ],
)
def test_invalid_fermi_energies(tmp_path, text, match):
    with pytest.raises(SymCWInputError, match=match):
        write_win(tmp_path, text)


# ==================================================
@pytest.mark.parametrize(
    "func, extra",
    [
        ("berry_get_kubo", {}),
        ("berry_get_kubo_tb", {}),
        ("berry_get_shc", {"shc_freq_scan": True}),
        ("spin_moment_main", {}),
    ],
)
def test_single_fermi_energy_required(func, extra):
    """as wannier90, the optical conductivity, the ac spin Hall conductivity and the spin moment reject a scan."""
    from symclosestwannier.analyzer import get_response

    cwi = {"num_fermi": 2, "fermi_energy": 0.0, "fermi_energy_list": [0.0, 0.5], "spin_decomp": False, **extra}
    with pytest.raises(SymCWInputError, match="needs a single Fermi energy"):
        getattr(get_response, func)(cwi, {})


# ==================================================
@pytest.mark.parametrize("func", ["berry_get_kubo", "berry_get_kubo_tb"])
def test_scan_of_one_energy_is_used(tmp_path, func):
    """
    a scan that rounds to one energy (nint(0.1) + 1) is that energy for the calculations at a single Fermi energy, as
    fermi_energy is: they use fermi_energy_list[0] (as wannier90), not fermi_energy = 0.
    """
    from symclosestwannier.analyzer import get_response
    from symclosestwannier.util.get_oper_R import get_v_R

    scan = write_win(tmp_path, "fermi_energy_min = 0.3\nfermi_energy_max = 0.4\nfermi_energy_step = 1.0\n")
    single = write_win(tmp_path, "fermi_energy = 0.3\n")
    zero = write_win(tmp_path, "")
    assert scan["num_fermi"] == 1 and scan["fermi_energy_list"] == [0.3] and scan["fermi_energy"] == 0.0

    model = Model()
    ops = model.operators_R()
    ops["v_R"] = get_v_R(make_cwi(model, False), ops["HH_R"])  # velocity of the tight-binding approximation
    kubo = {}
    for name, win in (("scan", scan), ("single", single), ("zero", zero)):
        cwi = {
            **make_cwi(model, False),
            **{k: win[k] for k in ("fermi_energy", "fermi_energy_list", "num_fermi")},
            "berry_kmesh": [3, 3, 3],
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
        }
        kubo[name] = np.array(getattr(get_response, func)(cwi, ops)[:2])  # Hermitian and anti-Hermitian parts

    np.testing.assert_allclose(kubo["scan"], kubo["single"], rtol=0, atol=0)
    for part in range(2):
        assert np.abs(kubo["scan"][part] - kubo["zero"][part]).max() > 1e-3 * np.abs(kubo["single"][part]).max()
