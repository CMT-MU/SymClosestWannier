"""
tests for the hr.dat reader used by the regression tests.
"""

import numpy as np
import pytest

from conftest import read_hr_dat

HEADER = "# test\n"


# ==================================================
def write(tmp_path, text):
    filename = tmp_path / "test_hr.dat"
    filename.write_text(HEADER + text)
    return filename


# ==================================================
def test_single_orbital_single_R(tmp_path):
    filename = write(tmp_path, "1\n1\n1\n0 0 0 1 1 -1.5 0.25\n")

    ndegen, irvec, Hr = read_hr_dat(filename)

    np.testing.assert_array_equal(ndegen, [1])
    np.testing.assert_array_equal(irvec, [[0, 0, 0]])
    np.testing.assert_array_equal(Hr, [[[-1.5 + 0.25j]]])


# ==================================================
def test_elements_placed_by_indices(tmp_path):
    rows = [
        "1 0 0 2 1 0.0 1.0",
        "1 0 0 1 1 3.0 0.0",
        "1 0 0 2 2 4.0 0.0",
        "1 0 0 1 2 0.0 -1.0",
        "0 0 0 1 1 1.0 0.0",
        "0 0 0 2 1 2.0 0.0",
        "0 0 0 1 2 2.0 0.0",
        "0 0 0 2 2 1.0 0.0",
    ]
    filename = write(tmp_path, "2\n2\n2 1\n" + "\n".join(rows) + "\n")

    ndegen, irvec, Hr = read_hr_dat(filename)

    np.testing.assert_array_equal(ndegen, [2, 1])
    np.testing.assert_array_equal(irvec, [[1, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(Hr[0], [[3.0, -1.0j], [1.0j, 4.0]])
    np.testing.assert_array_equal(Hr[1], [[1.0, 2.0], [2.0, 1.0]])


# ==================================================
@pytest.mark.parametrize(
    "rows",
    [
        # duplicated (R, m, n), (1, 2) missing.
        ["0 0 0 1 1 1.0 0.0", "0 0 0 2 1 2.0 0.0", "0 0 0 2 1 2.0 0.0", "0 0 0 2 2 1.0 0.0"],
        # orbital index out of range.
        ["0 0 0 1 1 1.0 0.0", "0 0 0 2 1 2.0 0.0", "0 0 0 1 3 2.0 0.0", "0 0 0 2 2 1.0 0.0"],
        # fractional orbital index.
        ["0 0 0 1 1 1.0 0.0", "0 0 0 2 1 2.0 0.0", "0 0 0 1.5 2 2.0 0.0", "0 0 0 2 2 1.0 0.0"],
        # fractional R vector.
        ["0 0 0 1 1 1.0 0.0", "0 0 0 2 1 2.0 0.0", "0 0.5 0 1 2 2.0 0.0", "0 0 0 2 2 1.0 0.0"],
        # inconsistent R label within a block.
        ["0 0 0 1 1 1.0 0.0", "0 0 0 2 1 2.0 0.0", "1 0 0 1 2 2.0 0.0", "0 0 0 2 2 1.0 0.0"],
    ],
)
def test_corrupted_indices_are_rejected(tmp_path, rows):
    filename = write(tmp_path, "2\n1\n1\n" + "\n".join(rows) + "\n")

    with pytest.raises(AssertionError):
        read_hr_dat(filename)
