"""
utility codes.
"""

import numpy as np


# ==================================================
def iterate_nd(size, pm=False):
    a = -size[0] if pm else 0
    b = size[0] + 1 if pm else size[0]
    if len(size) == 1:
        return np.array([(i,) for i in range(a, b)])
    else:
        return np.array([(i,) + tuple(j) for i in range(a, b) for j in iterate_nd(size[1:], pm=pm)])


# ==================================================
def iterate3dpm(size):
    assert len(size) == 3
    return iterate_nd(size, pm=True)


# ==================================================
def wigner_seitz(A, mp_grid, prec=1.0e-5):
    ws_search_size = np.array([1] * 3)
    dist_dim = np.prod((ws_search_size + 1) * 2 + 1)
    origin = divmod((dist_dim + 1), 2)[0] - 1
    real_metric = A.dot(A.T)
    mp_grid = np.array(mp_grid)
    irvec = []
    ndegen = []
    for n in iterate3dpm(mp_grid * ws_search_size):
        dist = []
        for i in iterate3dpm((1, 1, 1) + ws_search_size):
            ndiff = n - i * mp_grid
            dist.append(ndiff.dot(real_metric.dot(ndiff)))
        dist_min = np.min(dist)
        if abs(dist[origin] - dist_min) < prec:
            irvec.append(n)
            ndegen.append(np.sum(abs(dist - dist_min) < prec))

    irvec = np.array(irvec)
    ndegen = np.array(ndegen)

    return irvec, ndegen
