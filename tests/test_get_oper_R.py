"""
tests for real-space operators built by util/get_oper_R.py.
"""

import numpy as np

from symclosestwannier.util.get_oper_R import get_CC_R


# ==================================================
class _Nnkp:
    def __init__(self, kb2k, bveck):
        self._kb2k = kb2k
        self._bveck = bveck

    def kb2k(self):
        return self._kb2k

    def bveck(self):
        return self._bveck


class _CWInfo(dict):
    pass


# ==================================================
def make_cwi(seed=0):
    """
    artificial uHu data, H_{mn}(k,b1,b2) = <u_m(k+b1)|H(k)|u_n(k+b2)>, built as V(k,b1)^† M(k) V(k,b2) with hermitian M(k),
    so that H(k,b1,b2)^† = H(k,b2,b1) as for a real seedname.uHu.
    """
    rng = np.random.default_rng(seed)
    num_k, num_b, num_bands, num_wann, dim = 4, 6, 3, 2, 5

    kpoints = np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]])
    irvec = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    kb2k = rng.integers(0, num_k, size=(num_k, num_b))
    bveck = rng.normal(size=(num_k, num_b, 3))
    wb = rng.uniform(0.5, 1.5, size=num_b)

    Uk = rng.normal(size=(num_k, num_bands, num_wann)) + 1j * rng.normal(size=(num_k, num_bands, num_wann))
    V = rng.normal(size=(num_k, num_b, dim, num_bands)) + 1j * rng.normal(size=(num_k, num_b, dim, num_bands))
    M = rng.normal(size=(num_k, dim, dim)) + 1j * rng.normal(size=(num_k, dim, dim))
    M = M + M.transpose(0, 2, 1).conj()
    Hkb1b2 = np.einsum("kbxm,kxy,kdyn->kbdmn", V.conj(), M, V, optimize=True)

    cwi = _CWInfo(
        kpoints=kpoints,
        irvec=irvec,
        wb=wb,
        Uk=Uk,
        Hkb1b2=Hkb1b2,
        tb_gauge=False,
    )
    cwi.nnkp = _Nnkp(kb2k, bveck)

    return cwi


# ==================================================
def test_CC_R_keeps_antihermitian_part():
    """
    CC_ab(k) = <del_a u|H|del_b u> is not hermitian for a != b; only CC_ba(k) = CC_ab(k)^† holds (wannier90 get_oper.F90).
    Its antihermitian part gives LLambda = i(CC_ab - CC_ab^†) of the orbital magnetization and the gyrotropic response.
    """
    cwi = make_cwi()
    kb2k, bveck, wb = cwi.nnkp.kb2k(), cwi.nnkp.bveck(), cwi["wb"]
    Uk, Hkb1b2 = cwi["Uk"], cwi["Hkb1b2"]

    # CC_ab(k) = sum_{b1,b2} w_b1 w_b2 b1_a b2_b U(k+b1)^† H(k,b1,b2) U(k+b2)
    num_k = len(cwi["kpoints"])
    num_wann = Uk.shape[2]
    CC_k = np.zeros((3, 3, num_k, num_wann, num_wann), dtype=complex)
    for k in range(num_k):
        for b1 in range(len(wb)):
            for b2 in range(len(wb)):
                H = Uk[kb2k[k, b1]].conj().T @ Hkb1b2[k, b1, b2] @ Uk[kb2k[k, b2]]
                CC_k[:, :, k] += wb[b1] * wb[b2] * np.einsum("a,b,mn->abmn", bveck[k, b1], bveck[k, b2], H)

    phase = np.exp(-2j * np.pi * cwi["kpoints"] @ cwi["irvec"].T)
    CC_R_ref = np.einsum("kR,abkmn->abRmn", phase, CC_k) / num_k

    CC_R = get_CC_R(cwi)

    np.testing.assert_allclose(CC_R, CC_R_ref, rtol=0, atol=1e-10)

    LLambda = 1.0j * (CC_k[0, 1] - CC_k[0, 1].transpose(0, 2, 1).conj())
    assert np.max(np.abs(LLambda)) > 1.0
