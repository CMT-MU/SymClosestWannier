"""
berry phase and position operator.
"""
import numpy as np


# ==================================================
def Berry(cwi):
    """
    berry phase and position operator.

    Args:
        cwi (SystemInfo): CWInfo.
    """
    nnkp = cwi.nnkp
    mmn = cwi.mmn
    Uk = cwi.umat["Uk"]

    num_k = nnkp["num_k"]
    num_wann = nnkp["num_wann"]

    kpoints = np.array(nnkp["kpoints"])
    bveck = nnkp.bveck()
    wk = nnkp.wk()
    kb2k = nnkp.kb2k()

    Mkb = np.array(mmn["Mkb"])

    ### Unitary transform Mkb ###
    Uk = np.array(Uk)
    Mkb_w = np.einsum("klm, kblp, kbpn->kbmn", np.conj(Uk), Mkb, Uk[kb2k[:, :], :, :], optimize=True)  # Eq. (61)

    irvec = cwi["irvec"]

    kr = np.einsum("ka,ra->kr", kpoints, irvec, optimize=True)
    fac = np.exp(-2 * np.pi * 1j * kr)

    #
    # position operator
    #
    # <wik|r|wjk>
    Awkmn = 1j * np.einsum("kb,kba,kbmn->akmn", wk, bveck, Mkb_w, optimize=True)
    Awknn = -1 * np.einsum("kb,kba,kbnn->akn", wk, bveck, np.imag(np.log(Mkb_w)), optimize=True)
    np.einsum("aknn->akn", Awkmn)[:] = Awknn
    Awkmn = 0.5 * (Awkmn + np.einsum("akmn->aknm", Awkmn.conj()))

    # <wi0|r|wjR>
    rr_w = np.einsum("kr,akmn->armn", fac, Awkmn, optimize=True) / num_k

    #
    # berry phase
    #
    # i<wik|∇wjk>
    ak_w = 1j * np.einsum("kb,kba,kbmn->akmn", wk, bveck, (Mkb_w - np.eye(num_wann)), optimize=True)
    ak_w = 0.5 * (ak_w + np.einsum("akmn->aknm", ak_w.conj()))

    # <wi0|a|wjR>
    ar_w = np.einsum("akmn,kr->armn", ak_w, fac, optimize=True) / num_k

    return rr_w, ar_w
