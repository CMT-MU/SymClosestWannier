"""
symmetrization with atoms outside the unit cell in seedname.nnkp: MultiPie puts the sites in [0,1), and the lattice vectors R
of H(R) are shifted by the lattice vector between the two positions before the SAMB decomposition, so that the result does
not depend on where the atoms are put.
"""

import os
import shutil

import numpy as np
import pytest

from symclosestwannier.cw.cw_creator import cw_creator
from symclosestwannier.cw.cw_model import CWModel
from symclosestwannier.util.exceptions import SymCWInputError
from symclosestwannier.util.utility import construct_Or, shift_matrix_dict_r, site_offset, site_offsets

from conftest import EXAMPLES_DIR, read_hr_dat
from test_symmetrization import eigenvalues, read_hr_sym_dat

SEEDNAME = "graphene_pz"
# the second projection (C_1 of the model at [1/3,2/3,0]) is moved by this lattice vector in seedname.win/nnkp/amn
# (in the plane: along c, the k grid has only k3 = 0, and a shift by c does not change seedname.amn).
SHIFT = np.array([1, -1, 0])
# the first projection, [2/3,1/3,0], is C_2 of the MultiPie model.
KET_AMN = "[[C,2,1,pz],[C,1,1,pz]]"


# ==================================================
def test_site_offset():
    A = np.diag([3.0, 3.0, 30.0])
    A_model = np.diag([3.0, 3.0, 1000.0])
    assert site_offset([0.25, -0.75, 0.0], [0.25, 0.25, 0.0]) == [0, -1, 0]
    assert site_offset([0.25, 0.25, 0.5], [0.25, 0.25, 0.0]) is None
    # vacuum direction of a slab: the same distance (-1.5 A) along a3 of different lengths.
    assert site_offset([0.0, 0.0, -0.05], [0.0, 0.0, 0.9985], A, A_model) == [0, 0, -1]
    assert site_offset([0.0, 0.0, 0.05], [0.0, 0.0, 0.0015], A, A_model) == [0, 0, 0]
    assert site_offset([0.0, 0.0, 0.05], [0.0, 0.0, 0.0015]) is None
    assert site_offset([0.0, 0.0, 0.06], [0.0, 0.0, 0.0015], A, A_model) is None


# ==================================================
def test_site_offsets():
    """
    a model translated as a whole: MoS2 with Mo at z = 0 and S at z = +-1.5735 A (c = 31.66 A) in seedname.nnkp,
    and all the atoms shifted by c/2 in the model (c = 1000 A).
    """
    A = np.diag([3.166, 3.166, 31.66])
    A_model = np.diag([3.166, 3.166, 1000.0])
    pos = [[2 / 3, 1 / 3, 0.0], [1 / 3, 2 / 3, 0.0497], [1 / 3, 2 / 3, -0.0497], [1 / 3, 2 / 3 - 1, -0.0497]]
    pos_model = [[2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.5015735], [1 / 3, 2 / 3, 0.4984265], [1 / 3, 2 / 3, 0.4984265]]
    assert site_offsets(pos, pos_model, A, A_model) == ([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, -1, 0]], None)
    assert site_offsets(pos, pos_model, A, A_model, translation=False) == (None, 0)
    # the model with the positions of seedname.nnkp: the S below Mo is at z ~ 1 in MultiPie.
    pos_model = [[2 / 3, 1 / 3, 0.0], [1 / 3, 2 / 3, 0.0015735], [1 / 3, 2 / 3, 0.9984265], [1 / 3, 2 / 3, 0.9984265]]
    assert site_offsets(pos, pos_model, A, A_model, translation=False) == ([[0, 0, 0], [0, 0, 0], [0, 0, -1], [0, -1, -1]], None)
    # not a common translation.
    assert site_offsets(pos[:2], [[2 / 3, 1 / 3, 0.0], [2 / 3, 1 / 3, 0.0015735]], A, A_model) == (None, 1)


# ==================================================
def test_shift_matrix_dict_r():
    """
    shifting R and constructing the operator back with the same offset is the identity.
    """
    rng = np.random.default_rng(0)
    irvec = [(n1, n2, 0) for n1 in range(-2, 3) for n2 in range(-2, 3)]
    Or = rng.random((len(irvec), 2, 2))
    offset = [[0, 0, 0], [1, -1, 0]]

    Or_dict = shift_matrix_dict_r(CWModel.matrix_dict_r(Or, irvec), offset)
    assert Or_dict[(1, -1, 0, 0, 1)] == Or[irvec.index((0, 0, 0)), 0, 1]
    assert Or_dict[(-1, 1, 0, 1, 0)] == Or[irvec.index((0, 0, 0)), 1, 0]
    assert Or_dict[(0, 0, 0, 1, 1)] == Or[irvec.index((0, 0, 0)), 1, 1]

    # each matrix element as a "SAMB" with coefficient 1.
    samb = {f"z{i}": {k: 1.0} for i, k in enumerate(Or_dict)}
    coeff = {f"z{i}": v for i, v in enumerate(Or_dict.values())}
    np.testing.assert_allclose(construct_Or(coeff, 2, irvec, samb, offset), Or)


# ==================================================
def shift_inputs(workdir, seedname, n, shift):
    """
    move the projection centre of the n-th Wannier function (and the n-th atom of seedname.win) by a lattice vector.

    The trial orbital g_n(r - T) gives A_mn(k) exp(-2πi k.T), with k and T in fractional coordinates.
    """
    # seedname.nnkp
    nnkp = workdir / f"{seedname}.nnkp"
    lines = nnkp.read_text().splitlines(keepends=True)
    i = next(i for i, line in enumerate(lines) if line.strip().lower() == "begin projections")
    kpoints_start = next(j for j, line in enumerate(lines) if line.strip().lower() == "begin kpoints")
    num_k = int(lines[kpoints_start + 1])
    kpoints = np.array([[float(x) for x in lines[kpoints_start + 2 + j].split()] for j in range(num_k)])

    line = lines[i + 2 + 2 * n].split()
    pos = np.array([float(x) for x in line[:3]]) + shift
    lines[i + 2 + 2 * n] = " ".join([f"{x:.5f}" for x in pos] + line[3:]) + "\n"
    nnkp.write_text("".join(lines))

    # seedname.win
    win = workdir / f"{seedname}.win"
    lines = win.read_text().splitlines(keepends=True)
    i = next(i for i, line in enumerate(lines) if line.strip().lower() == "begin atoms_frac")
    line = lines[i + 1 + n].split()
    pos = np.array([float(x) for x in line[1:4]]) + shift
    lines[i + 1 + n] = " ".join([line[0]] + [repr(float(x)) for x in pos]) + "\n"
    win.write_text("".join(lines))

    # seedname.amn
    amn = workdir / f"{seedname}.amn"
    lines = amn.read_text().splitlines(keepends=True)
    for j in range(2, len(lines)):
        m, wf, ik, re, im = lines[j].split()
        if int(wf) == n + 1:
            v = complex(float(re), float(im)) * np.exp(-2j * np.pi * kpoints[int(ik) - 1] @ shift)
            lines[j] = f"{m:>5}{wf:>5}{ik:>5}  {v.real:.12f}  {v.imag:.12f}\n"
    amn.write_text("".join(lines))


# ==================================================
def read_z(filename):
    """
    read the SAMB coefficients, {zj: coeff}, from seedname_z.dat.cw.
    """
    z = {}
    for line in open(filename):
        v = line.split()
        if len(v) == 4 and not line.lstrip().startswith("#"):
            z[v[1]] = float(v[3])
    return z


# ==================================================
@pytest.fixture
def run_case(multipie_model, make_case, tmp_path):
    mpdir = multipie_model(os.path.join(EXAMPLES_DIR, SEEDNAME, f"{SEEDNAME}.py"), SEEDNAME)

    def _run(dirname, ket_amn, shift=None):
        extra = f"write_hr = true\nwrite_info_data = true\nsymmetrization = true\nmp_outdir = ./{SEEDNAME}\nmp_seedname = {SEEDNAME}\nket_amn = {ket_amn}"
        workdir = make_case(SEEDNAME, extra=extra, dirname=dirname)
        (workdir / SEEDNAME).mkdir()
        shutil.copy(mpdir / SEEDNAME / f"{SEEDNAME}.pkl", workdir / SEEDNAME)
        if shift is not None:
            shift_inputs(workdir, SEEDNAME, 1, shift)

        cw_creator(SEEDNAME)

        info, data, _ = CWModel.read_info_data(str(workdir / f"{SEEDNAME}.hdf5"))
        data = {"z": read_z(workdir / SEEDNAME / f"{SEEDNAME}_z.dat.cw"), "Ek_MAE_grid": data["Ek_MAE_grid"]}
        hr = read_hr_dat(str(workdir / f"{SEEDNAME}_hr.dat.cw"))
        hr_sym = read_hr_sym_dat(workdir / SEEDNAME / f"{SEEDNAME}_hr_sym.dat.cw", tmp_path / dirname)
        return info, data, hr, hr_sym

    return _run


# ==================================================
@pytest.mark.parametrize("ket_amn", ["auto", KET_AMN])
def test_shifted_atom(run_case, ket_amn):
    """
    the symmetrized model does not depend on the lattice vector by which an atom is moved in seedname.nnkp.
    """
    info0, data0, hr0, hr_sym0 = run_case("ref", ket_amn)
    info, data, hr, hr_sym = run_case("shifted", ket_amn, shift=SHIFT)

    assert info["ket_amn"] == info0["ket_amn"] == [["C", 2, 1, "pz"], ["C", 1, 1, "pz"]]
    np.testing.assert_array_equal(info0["site_offset"], [[0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(info["site_offset"], [[0, 0, 0], SHIFT])

    # the CW model itself only changes the gauge: the same bands on the k grid of the DFT calculation (19x19x1).
    # (off the grid, they differ by the different folding of H(R) into the Wigner-Seitz supercell.)
    k_grid = np.array([[i / 19, j / 19, 0] for i in range(19) for j in range(19)])
    np.testing.assert_allclose(eigenvalues(*hr, k_grid), eigenvalues(*hr0, k_grid), rtol=0, atol=1e-6)
    k = np.random.default_rng(0).random((20, 3))

    # the same SAMB coefficients and symmetrized bands.
    z0, z = data0["z"], data["z"]
    assert z.keys() == z0.keys()
    np.testing.assert_allclose([z[k_] for k_ in z], [z0[k_] for k_ in z], rtol=0, atol=1e-8)
    assert np.max(np.abs([z0[k_] for k_ in z])) > 0.1
    np.testing.assert_allclose(eigenvalues(*hr_sym, k), eigenvalues(*hr_sym0, k), rtol=0, atol=1e-8)
    np.testing.assert_allclose(data["Ek_MAE_grid"], data0["Ek_MAE_grid"], rtol=0, atol=1e-8)


# ==================================================
def test_wrong_ket_amn(run_case):
    """
    an explicit ket_amn whose site is not at the projection centre modulo the lattice vectors is an error.
    """
    with pytest.raises(SymCWInputError, match="does not match the position"):
        run_case("wrong", "[[C,1,1,pz],[C,2,1,pz]]")
