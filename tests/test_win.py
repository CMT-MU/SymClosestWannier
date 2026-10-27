"""
reading of seedname.win (Win).
"""

import pytest

from symclosestwannier.cw.win import Win
from symclosestwannier.util.exceptions import SymCWInputError

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
    (tmp_path / "seed.win").write_text("num_bands = 2\nnum_wann = 2\n" + text + _cell)
    return Win(str(tmp_path), "seed")


# ==================================================
def test_fortran_exponent(tmp_path):
    win = write_win(tmp_path, "dis_froz_max = 30.0d0\ndis_froz_min = -8.D-1\ndis_win_max = 1.5e+2\ndis_win_min = .5D1\n")

    assert win["dis_froz_max"] == 30.0
    assert win["dis_froz_min"] == -0.8
    assert win["dis_win_max"] == 150.0
    assert win["dis_win_min"] == 5.0


# ==================================================
@pytest.mark.parametrize("value", ["1d", "1d+", "1d2d3"])
def test_fortran_exponent_malformed(tmp_path, value):
    with pytest.raises(ValueError):
        write_win(tmp_path, f"dis_froz_max = {value}\n")


# ==================================================
@pytest.mark.parametrize("text, expected", [("", False), ("spn_formatted = .true.\n", True), ("spn_formatted = false\n", False)])
def test_spn_formatted(tmp_path, text, expected):
    assert write_win(tmp_path, text)["spn_formatted"] is expected


# ==================================================
@pytest.mark.parametrize("text, expected", [("", False), ("spn_formatted = .true.\n", True)])
def test_spn_formatted_passed_to_spn(make_case, monkeypatch, text, expected):
    from symclosestwannier.cw import cw_info
    from symclosestwannier.cw.spn import Spn

    calls = []

    class RecordingSpn(Spn):
        def __init__(self, topdir=None, seedname="cwannier", formatted=False, dic=None):
            calls.append(formatted)
            dict.__init__(self)

    monkeypatch.setitem(cw_info._class_map, "spn", RecordingSpn)
    workdir = make_case("graphene_pz")
    with open(workdir / "graphene_pz.win", "a") as fp:
        fp.write("\nspin_moment = true\n" + text)

    cw_info.CWInfo(str(workdir), "graphene_pz")

    assert calls == [expected]


# ==================================================
_shc = "berry = true\nberry_task = shc\nshc_bandshift = true\n"


# ==================================================
@pytest.mark.parametrize("value", ["1", "3"])
def test_shc_bandshift_firstband(tmp_path, value):
    win = write_win(tmp_path, _shc + f"shc_bandshift_firstband = {value}\n")
    assert win["shc_bandshift"] is True
    assert win["shc_bandshift_firstband"] == int(value)
    assert isinstance(win["shc_bandshift_firstband"], int)


# ==================================================
@pytest.mark.parametrize(
    "text",
    [
        _shc,
        _shc + "shc_bandshift_firstband = 0\n",
        "shc_bandshift_firstband = 0\n",  # a supplied value is checked even if not used, as in wannier90
    ],
)
def test_shc_bandshift_firstband_invalid(tmp_path, text):
    with pytest.raises(SymCWInputError, match="shc_bandshift_firstband"):
        write_win(tmp_path, text)


# ==================================================
@pytest.mark.parametrize("text", ["shc_bandshift = true\n", "berry = true\nberry_task = ahc\nshc_bandshift = true\n"])
def test_shc_bandshift_inactive(tmp_path, text):
    """
    without the spin Hall conductivity the shift is off, and shc_bandshift_firstband is not required.
    """
    assert write_win(tmp_path, text)["shc_bandshift"] is False
