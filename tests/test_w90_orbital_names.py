"""
the orbital names that convert_w90_orbital gives to Wannier90 projections must be the names of MultiPie's model kets,
since sort_ket_matrix_dict looks every Wannier function up in the ket list of the SAMB model.
"""

import pytest

from symclosestwannier.util.utility import convert_w90_orbital

util_wannier = pytest.importorskip("multipie.util.util_wannier")

W90_M = {0: [1], 1: [1, 2, 3], 2: [1, 2, 3, 4, 5], 3: [1, 2, 3, 4, 5, 6, 7]}


# ==================================================
@pytest.mark.parametrize("l", [0, 1, 2, 3])
@pytest.mark.parametrize("s", [0, 1, -1])
def test_names_match_multipie(l, s):
    for m in W90_M[l]:
        assert convert_w90_orbital(l, m, 1, s) == util_wannier._convert_w90_orbital(l, m, 1, s)[1]


# ==================================================
def test_f_names():
    assert [convert_w90_orbital(3, m, 1, 0) for m in W90_M[3]] == ["faz", "f3x", "f3y", "fbz", "f3", "f2", "f1"]
