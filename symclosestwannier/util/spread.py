"""
utility codes for spreads (OmegaI, Omega_D, Omega_OD).
"""

import subprocess
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import gc

from symclosestwannier.util.utility import (
    fermi,
    fourier_transform_r_to_k,
    fourier_transform_r_to_k_vec,
    spin_zeeman_interaction,
)
from symclosestwannier.analyzer.get_response import utility_w0gauss

_num_proc = multiprocessing.cpu_count()


# ==================================================
def get_spreads(cwi):
    """
    spreads (OmegaI, Omega_D, Omega_OD).

    Args:
        cwi (CWInfo): CWInfo.

    Returns:
        ndarray: lindhard function.
    """
    Mkb = np.array(cwi["Mkb"])
    Uk = np.array(cwi["Uk"])

    kb2k = cwi.nnkp.kb2k()
    bveck = cwi.nnkp.bveck()
    wb = cwi["wb"]

    num_wann = cwi["num_wann"]
    num_k = cwi["num_k"]

    Mkb_w = np.einsum("klm, kblp, kbpn->kbmn", np.conj(Uk), Mkb, Uk[kb2k[:, :], :, :], optimize=True)

    # OmegaI
    Mkb_w2 = np.einsum("kbmn,kbmn->kb", Mkb_w, np.conj(Mkb_w), optimize=True).real
    OmegaI = np.einsum("b, kb->", wb, num_wann - Mkb_w2, optimize=True) / num_k

    # OmegaOD
    Mkb_w2_diag = np.einsum("kbnn,kbnn->kb", Mkb_w, np.conj(Mkb_w), optimize=True).real
    OmegaOD = np.einsum("b, kb->", wb, Mkb_w2 - Mkb_w2_diag, optimize=True) / num_k

    # OmegaD
    Mkb_w_diag = np.einsum("kbnn->kbn", Mkb_w, optimize=True)
    imln_Mkb_w_diag = np.log(Mkb_w_diag).imag
    r = -1.0 / num_k * np.einsum("b,ba,kbn->na", wb, bveck, imln_Mkb_w_diag, optimize=True)
    qn = imln_Mkb_w_diag + np.einsum("ba, na->bn", bveck, r, optimize=True)[np.newaxis, :, :]
    OmegaD = np.einsum("b, kbn->", wb, qn**2, optimize=True) / num_k

    return OmegaI, OmegaOD, OmegaD
