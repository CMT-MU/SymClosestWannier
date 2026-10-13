"""
tests for the selection of SAMBs by irreps in seedname.cwin.
"""

from pathlib import Path

import pytest

from symclosestwannier.cw.cw_model import CWModel

TUTORIAL01 = Path(__file__).resolve().parents[1] / "tutorials" / "tutorial01"


# ==================================================
@pytest.mark.parametrize(
    "irreps, expected",
    [
        ("all", {}),  # all irreps.
        ("full", {"Gamma": "IR"}),  # only the identity irrep, which keeps the full symmetry.
        (["A1", "E"], {"Gamma": ["A1", "E"]}),
    ],
)
def test_samb_select(irreps, expected):
    assert CWModel.samb_select(irreps) == expected


# ==================================================
def test_full_selects_only_identity_irrep():
    # the MultiPie model of CH4 (Td) in tutorial01, with SAMBs of A1, A2, E, T1 and T2.
    from multipie import MaterialModel

    mm = MaterialModel(topdir=str(TUTORIAL01), verbose=False)
    mm.load("ch4")

    def irreps(select):
        z = mm.get_combined_samb_matrix(fmt="value", digit=15, select=select, bond=False)
        return {mm["combined_id"][j][2][2] for j in z}, set(z)

    irreps_all, z_all = irreps(CWModel.samb_select("all"))
    irreps_full, z_full = irreps(CWModel.samb_select("full"))
    assert irreps_full == {"A1"}
    assert irreps_all > {"A1"}
    assert z_full and z_full < z_all
