"""
ket_samb_list reduces the kets of a MultiPie model to [atom, sublattice, rank, orbital]
for both ket formats: [atom, sublattice, rank, orbital] (MultiPie <= 2.2) and
[atom, sublattice, rank, idx, orbital] (MultiPie >= 2.3, 2026-07).
"""

from symclosestwannier.util.utility import ket_samb_list


def test_ket_samb_list_four_and_five_fields():
    four = {"full_matrix": {"ket": [["Ta", 1, 0, "s"], ["Ta", 1, 1, "px"], ["O", 2, 1, "pz"]]}}
    five = {"full_matrix": {"ket": [["Ta", 1, 0, 0, "s"], ["Ta", 1, 1, 0, "px"], ["O", 2, 1, 2, "pz"]]}}
    expected = [["Ta", 1, 0, "s"], ["Ta", 1, 1, "px"], ["O", 2, 1, "pz"]]
    assert ket_samb_list(four) == expected
    assert ket_samb_list(five) == expected
