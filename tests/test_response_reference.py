"""
the responses of postcw reproduce the reference values in tests/data/response_reference.npz (see response_reference.py),
computed for the reference model of reference_model.py by the implementation before its rewrite.
"""

import numpy as np
import pytest

from response_reference import CASES, FILE, compute

RTOL = 1e-7


# ==================================================
def _close(actual, desired, err_msg):
    scale = np.max(np.abs(desired)) if desired.size else 0.0
    np.testing.assert_allclose(actual, desired, rtol=RTOL, atol=RTOL * scale, err_msg=err_msg)


@pytest.mark.parametrize("name, entry, settings", CASES, ids=[c[0] for c in CASES])
def test_response_reproduces_reference(name, entry, settings):
    ref = np.load(FILE)
    expected = {k: ref[k] for k in ref.files if k.split("/", 1)[0] == name}
    assert expected, f"no reference values for {name}"

    actual = compute(name, entry, settings)
    assert sorted(actual) == sorted(expected)

    for key, desired in expected.items():
        value = actual[key]
        if desired.dtype.kind == "U":  # the keys whose value is None
            assert sorted(value) == sorted(desired), key
            continue
        assert value.shape == desired.shape, key
        # with tb_gauge, the terms J0, J1, J2 of ahc_list and morb* differ from those of the Wannier gauge, but they are
        # compared with reference values of the same gauge.
        _close(value, desired, key)


def test_shc_adaptive_kmesh_keeps_points_below_threshold():
    """
    the adaptive k refinement of the SHC Fermi-energy scan only changes the Fermi energies whose spin Berry curvature
    exceeds the threshold at some k point; the others (0.3 and 1.2 eV here) keep the plain mesh, also at the points
    refined for another Fermi energy.
    """
    cases = {c[0]: c for c in CASES}
    plain = compute(*cases["shc_fermi"])["shc_fermi/shc_fermi"]
    adpt = compute(*cases["shc_fermi_adpt_kmesh"])["shc_fermi_adpt_kmesh/shc_fermi"]

    np.testing.assert_allclose(adpt[1:3], plain[1:3], rtol=1e-12)
    assert np.all(np.abs(adpt[[0, 3]] - plain[[0, 3]]) > 1e-3 * np.abs(plain[[0, 3]]))

    # a threshold above every value: no point is refined
    name, entry, settings = cases["shc_fermi_adpt_kmesh"]
    none = compute(name, entry, dict(settings, berry_curv_adpt_kmesh_thresh=1.0e6))[f"{name}/shc_fermi"]
    np.testing.assert_allclose(none, plain, rtol=1e-12)


@pytest.mark.parametrize(
    "name, keys",
    [("ahc", ["ahc_list"]), ("morb", ["morb", "morb_LC", "morb_IC"]), ("kubo_gauss", ["kubo_H", "kubo_AH"])],
)
def test_totals_do_not_depend_on_tb_gauge(name, keys):
    """
    the AHC, the orbital magnetization (summed over the terms J0, J1, J2) and the optical conductivity do not depend on
    whether the orbital positions enter the Fourier phase (tb_gauge); independent of the reference values.
    """
    cases = {c[0]: c for c in CASES}
    _, entry, settings = cases[name]
    wannier = compute(name, entry, settings)
    tb = compute(name, entry, dict(settings, tb_gauge=True))
    for key in keys:
        a, b = wannier[f"{name}/{key}"], tb[f"{name}/{key}"]
        if key != "kubo_H" and key != "kubo_AH":
            a, b = a.sum(axis=1), b.sum(axis=1)
        _close(b, a, key)
