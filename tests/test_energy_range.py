"""
defaults of kubo_freq_max, kubo_eigval_max, gyrotropic_freq_max and gyrotropic_eigval_max, as wannier90
(w90_wannier90_readwrite_read_energy_range in postw90_readwrite.F90).
"""

import numpy as np
import pytest

from symclosestwannier.cw.win import Win, energy_range_defaults

KEYS = ("kubo_freq_max", "kubo_eigval_max", "gyrotropic_freq_max", "gyrotropic_eigval_max")

_cell = """
begin unit_cell_cart
ang
1 0 0
0 1 0
0 0 1
end unit_cell_cart
begin atoms_frac
A 0 0 0
end atoms_frac
"""


# ==================================================
def write_win(tmp_path, text):
    (tmp_path / "seed.win").write_text("num_bands = 4\nnum_wann = 2\n" + text + _cell)
    return Win(str(tmp_path), "seed")


# ==================================================
def test_not_given_is_none(tmp_path):
    win = write_win(tmp_path, "")
    assert all(win[k] is None for k in KEYS)


# ==================================================
def test_given_values_are_kept(tmp_path):
    text = "kubo_freq_max = 7.5\nkubo_eigval_max = 4.0\ngyrotropic_freq_max = 2.5\ngyrotropic_eigval_max = 3.0\n"
    win = write_win(tmp_path, text + "dis_froz_max = 1.0\n")
    energy_range_defaults(win)
    assert [win[k] for k in KEYS] == [7.5, 4.0, 2.5, 3.0]


# ==================================================
def test_frozen_window(tmp_path):
    """dis_froz_max - E_F + 0.6667 and dis_froz_max + 0.6667, with the first Fermi energy."""
    win = write_win(tmp_path, "dis_froz_max = 2.3\nfermi_energy = -0.2\n")
    win["Ek"] = [[-10.0, 0.0, 5.0, 20.0]]
    energy_range_defaults(win)
    assert win["kubo_freq_max"] == pytest.approx(2.3 + 0.2 + 0.6667)
    assert win["kubo_eigval_max"] == pytest.approx(2.3 + 0.6667)
    assert win["gyrotropic_freq_max"] == win["kubo_freq_max"]
    assert win["gyrotropic_eigval_max"] == win["kubo_eigval_max"]

    d = {"dis_froz_max": 2.3, "fermi_energy_list": [-1.0, -0.5, 0.0], "fermi_energy": None, **{k: None for k in KEYS}}
    energy_range_defaults(d)
    assert d["kubo_freq_max"] == pytest.approx(2.3 + 1.0 + 0.6667)


# ==================================================
def test_eigenvalues(tmp_path):
    """without a frozen window: max(Ek) - min(Ek) + 0.6667 and max(Ek) + 0.6667."""
    win = write_win(tmp_path, "")
    win["Ek"] = [[-10.0, 0.0], [5.0, 20.0]]
    energy_range_defaults(win)
    assert win["kubo_freq_max"] == pytest.approx(30.6667)
    assert win["kubo_eigval_max"] == pytest.approx(20.6667)


# ==================================================
def test_outer_window():
    """without a frozen window and eigenvalues: dis_win_max - dis_win_min + 0.6667 and dis_win_max + 0.6667."""
    d = {"dis_froz_max": 100000, "dis_win_min": -5.0, "dis_win_max": 3.0, **{k: None for k in KEYS}}
    energy_range_defaults(d)
    assert d["kubo_freq_max"] == pytest.approx(8.6667)
    assert d["kubo_eigval_max"] == pytest.approx(3.6667)


# ==================================================
def test_gyrotropic_defaults_do_not_follow_kubo(tmp_path):
    """as wannier90, the gyrotropic defaults come from the energies, not from kubo_freq_max given in seedname.win."""
    win = write_win(tmp_path, "kubo_freq_max = 9.0\nkubo_eigval_max = 9.5\n")
    win["Ek"] = [[0.0, 2.0]]
    energy_range_defaults(win)
    assert win["kubo_freq_max"] == 9.0
    assert win["gyrotropic_freq_max"] == pytest.approx(2.6667)
    assert win["gyrotropic_eigval_max"] == pytest.approx(2.6667)


# ==================================================
def test_postcw_default_frequencies(make_case):
    """
    postcw without kubo_freq_max: the graphene_pz example has dis_froz_max = 2.3 and fermi_energy = -0.1863, so the
    frequencies reach 2.3 + 0.1863 + 0.6667 eV (before, they stopped below the default 1.0 eV).
    """
    from symclosestwannier.analyzer.analyzer import analyzer
    from symclosestwannier.cw.cw_creator import cw_creator

    seedname = "graphene_pz"
    workdir = make_case(seedname, extra="write_info_data = true\nuse_tb_approximation = true\nverbose = false")
    with open(workdir / f"{seedname}.win", "a") as fp:
        fp.write("\nberry = true\nberry_task = kubo\nberry_kmesh = 6 6 1\nkubo_freq_step = 0.5\n")
        fp.write("kubo_smr_fixed_en_width = 0.2\n")

    cw_creator(seedname)
    analyzer(seedname)

    freq = np.loadtxt(workdir / f"{seedname}-kubo_S_xx.dat")[:, 0]
    freq_max = 2.3 + 0.1863 + 0.6667
    # the last frequency is kubo_freq_max, or the last one below it with frequency lists that leave it out
    assert freq_max - 0.5 - 1e-9 < freq.max() <= freq_max + 1e-9


# ==================================================
@pytest.mark.parametrize("value, frozen", [(150000.0, True), (100000.0, False)])
def test_inner_window_given(value, frozen):
    """the inner window is given unless dis_froz_max is the value of Win for an absent keyword (+100000)."""
    d = {"dis_froz_max": value, "fermi_energy_list": [0.0], "Ek": [[0.0, 2.0]], **{k: None for k in KEYS}}
    energy_range_defaults(d)
    assert d["kubo_eigval_max"] == pytest.approx(value + 0.6667 if frozen else 2.6667)


# ==================================================
@pytest.mark.parametrize(
    "fermi, expected",
    [
        ({"fermi_energy_list": np.array([-1.0, 0.0]), "fermi_energy": None}, -1.0),
        ({"fermi_energy_list": [], "fermi_energy": -0.5}, -0.5),
        ({"fermi_energy_list": None, "fermi_energy": None}, 0.0),
        ({}, 0.0),
    ],
)
def test_fermi_energy_of_the_inner_window(fermi, expected):
    d = {"dis_froz_max": 2.0, **fermi, **{k: None for k in KEYS}}
    energy_range_defaults(d)
    assert d["kubo_freq_max"] == pytest.approx(2.0 - expected + 0.6667)


# ==================================================
def test_all_four_and_nothing_to_use():
    d = {"Ek": [[-1.0, 3.0]], **{k: None for k in KEYS}}
    energy_range_defaults(d)
    assert [d[k] for k in KEYS] == pytest.approx([4.6667, 3.6667, 4.6667, 3.6667])

    d = {k: None for k in KEYS}
    energy_range_defaults(d)
    assert all(d[k] is None for k in KEYS)


# ==================================================
def test_response_sets_the_defaults(monkeypatch):
    """Response used without the analyzer of postcw (for example from a script) also sets them."""
    from symclosestwannier.analyzer import response

    monkeypatch.setattr(response.Response, "set_operators", lambda self: None)
    cwi = {"seedname": "model", "dis_froz_max": 100000, "Ek": [[0.0, 2.0]], **{k: None for k in KEYS}}
    response.Response(cwi, None)
    assert cwi["kubo_eigval_max"] == pytest.approx(2.6667)
    assert cwi["kubo_freq_max"] == pytest.approx(2.6667)


# ==================================================
@pytest.mark.parametrize("Ek", [[], np.zeros((0, 4))])
def test_empty_eigenvalues(Ek):
    """empty eigenvalues are not used: the outer window if given, otherwise nothing."""
    d = {"Ek": Ek, "dis_win_min": -5.0, "dis_win_max": 3.0, **{k: None for k in KEYS}}
    energy_range_defaults(d)
    assert d["kubo_freq_max"] == pytest.approx(8.6667)

    d = {"Ek": Ek, **{k: None for k in KEYS}}
    energy_range_defaults(d)
    assert all(d[k] is None for k in KEYS)


# ==================================================
def test_response_frozen_window_and_given_values(monkeypatch):
    """Response with an inner window, a Fermi energy without a list, and one value given."""
    from symclosestwannier.analyzer import response

    monkeypatch.setattr(response.Response, "set_operators", lambda self: None)
    cwi = {"seedname": "model", "dis_froz_max": 2.0, "fermi_energy": -1.0, **{k: None for k in KEYS}, "kubo_freq_max": 5.0}
    response.Response(cwi, None)
    assert cwi["kubo_freq_max"] == 5.0
    assert cwi["gyrotropic_freq_max"] == pytest.approx(2.0 + 1.0 + 0.6667)
    assert cwi["kubo_eigval_max"] == pytest.approx(2.6667)
