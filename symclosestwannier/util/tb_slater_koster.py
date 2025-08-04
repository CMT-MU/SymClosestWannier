"""
Slater-Koster parameters
"""

import sympy as sp

from multipie.multipole.util.spin_orbital_basis import _standard_spinless_basis

# ==================================================
"""
Slater-Koster independent paramters
"""
_SKP = {
    ("s", "s", "S"): sp.Symbol("V_{ss\sigma}", real=True),
    ("s", "p", "S"): sp.Symbol("V_{sp\sigma}", real=True),
    ("p", "p", "S"): sp.Symbol("V_{pp\sigma}", real=True),
    ("p", "p", "P"): sp.Symbol("V_{pp\pi}", real=True),
    ("s", "d", "S"): sp.Symbol("V_{sd\sigma}", real=True),
    ("s", "f", "S"): sp.Symbol("V_{sf\sigma}", real=True),
    ("p", "d", "S"): sp.Symbol("V_{pd\sigma}", real=True),
    ("p", "d", "P"): sp.Symbol("V_{pd\pi}", real=True),
    ("p", "f", "S"): sp.Symbol("V_{pf\sigma}", real=True),
    ("p", "f", "P"): sp.Symbol("V_{pf\pi}", real=True),
    ("d", "d", "S"): sp.Symbol("V_{dd\sigma}", real=True),
    ("d", "d", "P"): sp.Symbol("V_{dd\pi}", real=True),
    ("d", "d", "D"): sp.Symbol("V_{dd\delta}", real=True),
    ("d", "f", "S"): sp.Symbol("V_{df\sigma}", real=True),
    ("d", "f", "P"): sp.Symbol("V_{df\pi}", real=True),
    ("d", "f", "D"): sp.Symbol("V_{df\delta}", real=True),
    ("f", "f", "S"): sp.Symbol("V_{ff\sigma}", real=True),
    ("f", "f", "P"): sp.Symbol("V_{ff\pi}", real=True),
    ("f", "f", "D"): sp.Symbol("V_{ff\delta}", real=True),
    ("f", "f", "Phi"): sp.Symbol("V_{ff\phi}", real=True),
}


# ==================================================
def sk_parameter(o1, o2, l, m, n):
    """
    Slater-Koster parameters list

    r/|r| = (l,m,n)

    Args:
        o1 (str): <o1|
        o2 (str): |o2>
        l (sympy/float):
        m (sympy/float):
        n (sympy/float):
    """
    idx1, idx2 = _standard_spinless_basis["cubic"].index(o1), _standard_spinless_basis["cubic"].index(o2)
    if idx1 > idx2:
        o1, o2 = o2, o1

    a = sp.sqrt(l**2 + m**2 + n**2)
    l, m, n = l / a, m / a, n / a

    if (o1, o2) == ("s", "s"):
        return _SKP[("s", "s", "S")]
    elif (o1, o2) == ("s", "px"):
        return l * _SKP[("s", "p", "S")]
    elif (o1, o2) == ("s", "py"):
        return m * _SKP[("s", "p", "S")]
    elif (o1, o2) == ("s", "pz"):
        return n * _SKP[("s", "p", "S")]
    elif (o1, o2) == ("s", "du"):
        return (n**2 - sp.S(l**2 + m**2) / 2) * _SKP[("s", "d", "S")]
    elif (o1, o2) == ("s", "dv"):
        return sp.sqrt(3) / 2 * (l**2 - m**2) * _SKP[("s", "d", "S")]
    elif (o1, o2) == ("s", "dyz"):
        return sp.sqrt(3) * m * n * _SKP[("s", "d", "S")]
    elif (o1, o2) == ("s", "dzx"):
        return sp.sqrt(3) * n * l * _SKP[("s", "d", "S")]
    elif (o1, o2) == ("s", "dxy"):
        return sp.sqrt(3) * l * m * _SKP[("s", "d", "S")]
    elif (o1, o2) == ("s", "fxyz"):
        return sp.sqrt(15) * l * m * _SKP[("s", "f", "S")]
    elif (o1, o2) == ("s", "fax"):
        return sp.S(1) / 2 * l * (5 * l**2 - 3) * _SKP[("s", "f", "S")]
    elif (o1, o2) == ("s", "fay"):
        return sp.S(1) / 2 * m * (5 * m**2 - 3) * _SKP[("s", "f", "S")]
    elif (o1, o2) == ("s", "faz"):
        return sp.S(1) / 2 * n * (5 * n**2 - 3) * _SKP[("s", "f", "S")]
    elif (o1, o2) == ("s", "fbx"):
        return sp.S(15) / 2 * l * (m**2 - n**2) * _SKP[("s", "f", "S")]
    elif (o1, o2) == ("px", "px"):
        return l**2 * _SKP[("p", "p", "S")] + (1 - l**2) * _SKP[("p", "p", "P")]
    elif (o1, o2) == ("py", "py"):
        return m**2 * _SKP[("p", "p", "S")] + (1 - m**2) * _SKP[("p", "p", "P")]
    elif (o1, o2) == ("pz", "pz"):
        return n**2 * _SKP[("p", "p", "S")] + (1 - n**2) * _SKP[("p", "p", "P")]
    elif (o1, o2) == ("px", "py"):
        return l * m * (_SKP[("p", "p", "S")] - _SKP[("p", "p", "P")])
    elif (o1, o2) == ("px", "pz"):
        return l * n * (_SKP[("p", "p", "S")] - _SKP[("p", "p", "P")])
    elif (o1, o2) == ("py", "pz"):
        return m * n * (_SKP[("p", "p", "S")] - _SKP[("p", "p", "P")])
    elif (o1, o2) == ("px", "du"):
        return (
            l * (n**2 - sp.S(l**2 + m**2) / 2) * _SKP[("p", "d", "S")] - sp.sqrt(3) * l * n**2 * _SKP[("p", "d", "P")]
        )
    elif (o1, o2) == ("py", "du"):
        return (
            m * (n**2 - sp.S(l**2 + m**2) / 2) * _SKP[("p", "d", "S")] - sp.sqrt(3) * m * n**2 * _SKP[("p", "d", "P")]
        )
    elif (o1, o2) == ("pz", "du"):
        return (
            n * (n**2 - sp.S(l**2 + m**2) / 2) * _SKP[("p", "d", "S")]
            + sp.sqrt(3) * n * (l**2 + m**2) * _SKP[("p", "d", "P")]
        )
    elif (o1, o2) == ("px", "dv"):
        return (
            sp.sqrt(3) / 2 * l * (l**2 - m**2) * _SKP[("p", "d", "S")] + l * (1 - l**2 + m**2) * _SKP[("p", "d", "P")]
        )
    elif (o1, o2) == ("py", "dv"):
        return (
            sp.sqrt(3) / 2 * m * (l**2 - m**2) * _SKP[("p", "d", "S")] - m * (1 + l**2 - m**2) * _SKP[("p", "d", "P")]
        )
    elif (o1, o2) == ("pz", "dv"):
        return sp.sqrt(3) / 2 * n * (l**2 - m**2) * _SKP[("p", "d", "S")] - n * (l**2 - m**2) * _SKP[("p", "d", "P")]
    elif (o1, o2) == ("px", "dyz"):
        return l * m * n * (sp.sqrt(3) * _SKP[("p", "d", "S")] - 2 * _SKP[("p", "d", "P")])
    elif (o1, o2) == ("px", "dzx"):
        return sp.sqrt(3) * l**2 * n * _SKP[("p", "d", "S")] + n * (1 - 2 * l**2) * _SKP[("p", "d", "P")]
    elif (o1, o2) == ("px", "dxy"):
        return sp.sqrt(3) * l**2 * m * _SKP[("p", "d", "S")] + m * (1 - 2 * l**2) * _SKP[("p", "d", "P")]
    elif (o1, o2) == ("py", "dyz"):
        return sp.sqrt(3) * m**2 * n * _SKP[("p", "d", "S")] + n * (1 - 2 * m**2) * _SKP[("p", "d", "P")]
    elif (o1, o2) == ("py", "dzx"):
        return l * m * n * (sp.sqrt(3) * _SKP[("p", "d", "S")] - 2 * _SKP[("p", "d", "P")])
    elif (o1, o2) == ("py", "dxy"):
        return sp.sqrt(3) * m**2 * l * _SKP[("p", "d", "S")] + l * (1 - 2 * m**2) * _SKP[("p", "d", "P")]
    elif (o1, o2) == ("pz", "dyz"):
        return sp.sqrt(3) * n**2 * m * _SKP[("p", "d", "S")] + m * (1 - 2 * n**2) * _SKP[("p", "d", "P")]
    elif (o1, o2) == ("pz", "dzx"):
        return sp.sqrt(3) * n**2 * l * _SKP[("p", "d", "S")] + l * (1 - 2 * n**2) * _SKP[("p", "d", "P")]
    elif (o1, o2) == ("pz", "dxy"):
        return l * m * n * (sp.sqrt(3) * _SKP[("p", "d", "S")] - 2 * _SKP[("p", "d", "P")])
    elif (o1, o2) == ("px", "fxyz"):
        return (
            sp.sqrt(15) * l**2 * m * n * _SKP[("p", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(2) * (3 * l**2 - 1) * m * n * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("px", "fax"):
        return (
            sp.S(1) / 2 * l**2 * (5 * l**2 - 3) * _SKP[("p", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * (5 * l**2 - 1) * (l**2 - 1) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("px", "fay"):
        return (
            sp.S(1) / 2 * l * m * (5 * m**2 - 3) * _SKP[("p", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * l * m * (5 * m**2 - 1) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("px", "faz"):
        return (
            sp.S(1) / 2 * l * n * (5 * n**2 - 3) * _SKP[("p", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * l * n * (5 * n**2 - 1) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("px", "fbx"):
        return (
            sp.S(15) / 2 * l**2 * (m**2 - n**2) * _SKP[("p", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * (3 * l**2 - 1) * (m**2 - n**2) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("px", "fby"):
        return (
            sp.S(15) / 2 * l * m * (n**2 - l**2) * _SKP[("p", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * l * m * (3 * (n**2 - l**2) + 2) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("px", "fbz"):
        return (
            sp.S(15) / 2 * l * n * (l**2 - m**2) * _SKP[("p", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * l * n * (3 * (l**2 - m**2) - 2) * _SKP[("p", "f", "P")]
        )
    ###
    elif (o1, o2) == ("py", "fxyz"):
        return (
            sp.sqrt(15) * l * m**2 * n * _SKP[("p", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(2) * (3 * m**2 - 1) * n * l * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("py", "fax"):
        return (
            sp.S(1) / 2 * m * l * (5 * l**2 - 3) * _SKP[("p", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * m * l * (5 * l**2 - 1) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("py", "fay"):
        return (
            sp.S(1) / 2 * m**2 * (5 * m**2 - 3) * _SKP[("p", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * (5 * m**2 - 1) * (m**2 - 1) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("py", "faz"):
        return (
            sp.S(1) / 2 * m * n * (5 * n**2 - 3) * _SKP[("p", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * m * n * (5 * n**2 - 1) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("py", "fbx"):
        return (
            sp.S(15) / 2 * m * l * (m**2 - n**2) * _SKP[("p", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * m * l * (3 * (m**2 - n**2) - 2) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("py", "fby"):
        return (
            sp.S(15) / 2 * m**2 * (n**2 - l**2) * _SKP[("p", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * (3 * m**2 - 1) * (n**2 - l**2) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("py", "fbz"):
        return (
            sp.S(15) / 2 * m * n * (l**2 - m**2) * _SKP[("p", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * m * n * (3 * (l**2 - m**2) + 2) * _SKP[("p", "f", "P")]
        )
    ###
    elif (o1, o2) == ("pz", "fxyz"):
        return (
            sp.sqrt(15) * l * m * n**2 * _SKP[("p", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(2) * (3 * n**2 - 1) * n * l * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("pz", "fax"):
        return (
            sp.S(1) / 2 * n * l * (5 * l**2 - 3) * _SKP[("p", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * n * l * (5 * l**2 - 1) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("pz", "fay"):
        return (
            sp.S(1) / 2 * n * m * (5 * m**2 - 3) * _SKP[("p", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * n * m * (5 * m**2 - 1) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("pz", "faz"):
        return (
            sp.S(1) / 2 * n**2 * (5 * n**2 - 3) * _SKP[("p", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * (5 * n**2 - 1) * (n**2 - 1) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("pz", "fbx"):
        return (
            sp.S(15) / 2 * n * l * (m**2 - n**2) * _SKP[("p", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * n * l * (3 * (m**2 - n**2) + 2) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("pz", "fby"):
        return (
            sp.S(15) / 2 * n * m * (n**2 - l**2) * _SKP[("p", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * n * m * (3 * (n**2 - l**2) - 2) * _SKP[("p", "f", "P")]
        )
    elif (o1, o2) == ("pz", "fbz"):
        return (
            sp.S(15) / 2 * n**2 * (l**2 - m**2) * _SKP[("p", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * (3 * n**2 - 1) * (l**2 - m**2) * _SKP[("p", "f", "P")]
        )
    ###
    elif (o1, o2) == ("du", "du"):
        return (
            (n**2 - sp.S(l**2 + m**2) / 2) ** 2 * _SKP[("d", "d", "S")]
            + 3 * n**2 * (l**2 + m**2) * _SKP[("d", "d", "P")]
            + sp.S(3) / 4 * (l**2 + m**2) ** 2 * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("du", "dv"):
        return (
            sp.sqrt(3) / 2 * (l**2 - m**2) * (n**2 - sp.S(l**2 + m**2) / 2) * _SKP[("d", "d", "S")]
            - sp.sqrt(3) * n**2 * (l**2 - m**2) * _SKP[("d", "d", "P")]
            + sp.sqrt(3) / 4 * (1 + n**2) * (l**2 - m**2) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("du", "dyz"):
        return (
            sp.sqrt(3) * m * n * (n**2 - sp.S(l**2 + m**2) / 2) * _SKP[("d", "d", "S")]
            + sp.sqrt(3) * m * n * (l**2 + m**2 - n**2) * _SKP[("d", "d", "P")]
            - sp.sqrt(3) / 2 * m * n * (l**2 + m**2) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("du", "dzx"):
        return (
            sp.sqrt(3) * l * n * (n**2 - sp.S(l**2 + m**2) / 2) * _SKP[("d", "d", "S")]
            + sp.sqrt(3) * l * n * (l**2 + m**2 - n**2) * _SKP[("d", "d", "P")]
            - sp.sqrt(3) / 2 * l * n * (l**2 + m**2) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("du", "dxy"):
        return (
            sp.sqrt(3) * l * m * (n**2 - sp.S(l**2 + m**2) / 2) * _SKP[("d", "d", "S")]
            - 2 * sp.sqrt(3) * l * m * n**2 * _SKP[("d", "d", "P")]
            + sp.sqrt(3) / 2 * l * m * (1 + n**2) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("dv", "dv"):
        return (
            sp.S(3) / 4 * (l**2 - m**2) ** 2 * _SKP[("d", "d", "S")]
            + ((l**2 + m**2) - (l**2 - m**2) ** 2) * _SKP[("d", "d", "P")]
            + (n**2 + (l**2 - m**2) ** 2 / 4) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("dv", "dyz"):
        return (
            sp.S(3) / 2 * m * n * (l**2 - m**2) * _SKP[("d", "d", "S")]
            - m * n * (1 + 2 * (l**2 - m**2)) * _SKP[("d", "d", "P")]
            + m * n * (1 + (l**2 - m**2) / 2) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("dv", "dzx"):
        return (
            sp.S(3) / 2 * l * n * (l**2 - m**2) * _SKP[("d", "d", "S")]
            + l * n * (1 - 2 * (l**2 - m**2)) * _SKP[("d", "d", "P")]
            - l * n * (1 - (l**2 - m**2) / 2) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("dv", "dxy"):
        return (
            sp.S(3) / 2 * l * m * (l**2 - m**2) * _SKP[("d", "d", "S")]
            - 2 * l * m * (l**2 - m**2) * _SKP[("d", "d", "P")]
            + sp.S(1) / 2 * l * m * (l**2 - m**2) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("dyz", "dyz"):
        return (
            3 * m**2 * n**2 * _SKP[("d", "d", "S")]
            + (m**2 + n**2 - 4 * m**2 * n**2) * _SKP[("d", "d", "P")]
            + (l**2 + n**2 * m**2) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("dzx", "dzx"):
        return (
            3 * n**2 * l**2 * _SKP[("d", "d", "S")]
            + (n**2 + l**2 - 4 * n**2 * l**2) * _SKP[("d", "d", "P")]
            + (m**2 + n**2 * l**2) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("dxy", "dxy"):
        return (
            3 * l**2 * m**2 * _SKP[("d", "d", "S")]
            + (l**2 + m**2 - 4 * l**2 * m**2) * _SKP[("d", "d", "P")]
            + (n**2 + l**2 * m**2) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("dyz", "dzx"):
        return (
            3 * l * m * n**2 * _SKP[("d", "d", "S")]
            + l * m * (1 - 4 * n**2) * _SKP[("d", "d", "P")]
            - l * m * (1 - n**2) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("dyz", "dxy"):
        return (
            3 * l * m**2 * n * _SKP[("d", "d", "S")]
            + l * n * (1 - 4 * m**2) * _SKP[("d", "d", "P")]
            - l * n * (1 - m**2) * _SKP[("d", "d", "D")]
        )
    elif (o1, o2) == ("dzx", "dxy"):
        return (
            3 * l**2 * m * n * _SKP[("d", "d", "S")]
            + m * n * (1 - 4 * l**2) * _SKP[("d", "d", "P")]
            - m * n * (1 - l**2) * _SKP[("d", "d", "D")]
        )
    ###
    elif (o1, o2) == ("dxy", "fxyz"):
        return (
            sp.sqrt(45) * l**2 * m**2 * n * _SKP[("d", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(2) * n * (6 * l**2 * m**2 + n**2 - 1) * _SKP[("d", "f", "P")]
            + n * (3 * l**2 * m**2 + 2 * n**2 - 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dxy", "fax"):
        return (
            sp.sqrt(3) / 2 * l**2 * m * (5 * l**2 - 3) * _SKP[("d", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * m * (5 * l**2 - 1) * (2 * l**2 - 1) * _SKP[("d", "f", "P")]
            + sp.sqrt(15) / 2 * l**2 * m * (l**2 - 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dxy", "fay"):
        return (
            sp.sqrt(3) / 2 * l * m**2 * (5 * m**2 - 3) * _SKP[("d", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * l * (5 * m**2 - 1) * (2 * m**2 - 1) * _SKP[("d", "f", "P")]
            + sp.sqrt(15) / 2 * l * m**2 * (m**2 - 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dxy", "faz"):
        return (
            sp.sqrt(3) / 2 * l * m * n * (5 * n**2 - 3) * _SKP[("d", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(2) * l * m * n * (5 * n**2 - 1) * _SKP[("d", "f", "P")]
            + sp.sqrt(15) / 2 * l * m * n * (n**2 + 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dxy", "fbx"):
        return (
            sp.S(3) / 2 * sp.sqrt(5) * l**2 * m * (m**2 - n**2) * _SKP[("d", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * m * ((6 * l**2 - 1) * (m**2 - n**2) - 2 * l**2) * _SKP[("d", "f", "P")]
            + sp.S(1) / 2 * m * (3 * l**2 * (m**2 - n**2) + 4 * n**2 - 2 * l**2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dxy", "fby"):
        return (
            sp.S(3) / 2 * sp.sqrt(5) * l * m**2 * (n**2 - l**2) * _SKP[("d", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * l * ((6 * m**2 - 1) * (n**2 - l**2) + 2 * m**2) * _SKP[("d", "f", "P")]
            + sp.S(1) / 2 * l * (3 * m**2 * (n**2 - l**2) - 4 * n**2 + 2 * m**2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dxy", "fbz"):
        return (
            sp.S(3) / 2 * sp.sqrt(5) * l * m * n * (l**2 - m**2) * _SKP[("d", "f", "S")]
            - 3 * sp.sqrt(5) / sp.sqrt(2) * l * m * n * (l**2 - m**2) * _SKP[("d", "f", "P")]
            + sp.S(3) / 2 * l * m * n * (l**2 - m**2) * _SKP[("d", "f", "D")]
        )
    ###
    elif (o1, o2) == ("dyz", "fxyz"):
        return (
            sp.sqrt(45) * m**2 * n**2 * l * _SKP[("d", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(2) * l * (6 * m**2 * n**2 + l**2 - 1) * _SKP[("d", "f", "P")]
            + l * (3 * m**2 * n**2 + 2 * l**2 - 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dyz", "fax"):
        return (
            sp.sqrt(3) / 2 * m * n * l * (5 * l**2 - 3) * _SKP[("d", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(2) * m * n * l * (5 * l**2 - 1) * _SKP[("d", "f", "P")]
            + sp.sqrt(15) / 2 * m * n * l * (l**2 + 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dyz", "fay"):
        return (
            sp.sqrt(3) / 2 * m**2 * n * (5 * m**2 - 3) * _SKP[("d", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * n * (5 * m**2 - 1) * (2 * m**2 - 1) * _SKP[("d", "f", "P")]
            + sp.sqrt(15) / 2 * m**2 * n * (m**2 - 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dyz", "faz"):
        return (
            sp.sqrt(3) / 2 * m * n**2 * (5 * n**2 - 3) * _SKP[("d", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * m * (5 * n**2 - 1) * (2 * n**2 - 1) * _SKP[("d", "f", "P")]
            + sp.sqrt(15) / 2 * m * n**2 * (n**2 - 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dyz", "fbx"):
        return (
            sp.S(3) / 2 * sp.sqrt(5) * m * n * l * (m**2 - n**2) * _SKP[("d", "f", "S")]
            - 3 * sp.sqrt(5) / sp.sqrt(2) * m * n * l * (m**2 - n**2) * _SKP[("d", "f", "P")]
            + sp.S(3) / 2 * m * n * l * (m**2 - n**2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dyz", "fby"):
        return (
            sp.S(3) / 2 * sp.sqrt(5) * m**2 * n * (n**2 - l**2) * _SKP[("d", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * n * ((6 * m**2 - 1) * (n**2 - l**2) - 2 * m**2) * _SKP[("d", "f", "P")]
            + sp.S(1) / 2 * n * (3 * m**2 * (n**2 - l**2) + 4 * l**2 - 2 * m**2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dyz", "fbz"):
        return (
            sp.S(3) / 2 * sp.sqrt(5) * m * n**2 * (l**2 - m**2) * _SKP[("d", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * m * ((6 * n**2 - 1) * (l**2 - m**2) + 2 * n**2) * _SKP[("d", "f", "P")]
            + sp.S(1) / 2 * m * (3 * n**2 * (l**2 - m**2) - 4 * l**2 + 2 * n**2) * _SKP[("d", "f", "D")]
        )
    ###
    elif (o1, o2) == ("dzx", "fxyz"):
        return (
            sp.sqrt(45) * n**2 * l**2 * m * _SKP[("d", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(2) * m * (6 * n**2 * l**2 + m**2 - 1) * _SKP[("d", "f", "P")]
            + m * (3 * n**2 * l**2 + 2 * m**2 - 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dzx", "fax"):
        return (
            sp.sqrt(3) / 2 * n * l**2 * (5 * l**2 - 3) * _SKP[("d", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * n * (5 * l**2 - 1) * (2 * l**2 - 1) * _SKP[("d", "f", "P")]
            + sp.sqrt(15) / 2 * n * l**2 * (l**2 - 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dzx", "fay"):
        return (
            sp.sqrt(3) / 2 * n * l * m * (5 * m**2 - 3) * _SKP[("d", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(2) * n * l * m * (5 * m**2 - 1) * _SKP[("d", "f", "P")]
            + sp.sqrt(15) / 2 * n * l * m * (m**2 + 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dzx", "faz"):
        return (
            sp.sqrt(3) / 2 * n**2 * l * (5 * n**2 - 3) * _SKP[("d", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * l * (5 * n**2 - 1) * (2 * n**2 - 1) * _SKP[("d", "f", "P")]
            + sp.sqrt(15) / 2 * n**2 * l * (n**2 - 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dzx", "fbx"):
        return (
            sp.S(3) / 2 * sp.sqrt(5) * n * l**2 * (m**2 - n**2) * _SKP[("d", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * n * ((6 * l**2 - 1) * (m**2 - n**2) + 2 * l**2) * _SKP[("d", "f", "P")]
            + sp.S(1) / 2 * n * (3 * l**2 * (m**2 - n**2) - 4 * m**2 + 2 * l**2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dzx", "fby"):
        return (
            sp.S(3) / 2 * sp.sqrt(5) * n * l * m * (n**2 - l**2) * _SKP[("d", "f", "S")]
            - 3 * sp.sqrt(5) / sp.sqrt(2) * n * l * m * (n**2 - l**2) * _SKP[("d", "f", "P")]
            + sp.S(3) / 2 * n * l * m * (n**2 - l**2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dzx", "fbz"):
        return (
            sp.S(3) / 2 * sp.sqrt(5) * n**2 * l * (l**2 - m**2) * _SKP[("d", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * l * ((6 * n**2 - 1) * (l**2 - m**2) - 2 * n**2) * _SKP[("d", "f", "P")]
            + sp.S(1) / 2 * l * (3 * n**2 * (l**2 - m**2) + 4 * m**2 - 2 * n**2) * _SKP[("d", "f", "D")]
        )
    ###
    elif (o1, o2) == ("dv", "fxyz"):
        return (
            sp.S(3) / 2 * sp.sqrt(5) * l * m * n * (l**2 - m**2) * _SKP[("d", "f", "S")]
            - 3 * sp.sqrt(5) / sp.sqrt(2) * l * m * n * (l**2 - m**2) * _SKP[("d", "f", "P")]
            + sp.S(3) / 2 * l * m * n * (l**2 - m**2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dv", "fax"):
        return (
            sp.sqrt(3) / 4 * l * (l**2 - m**2) * (5 * l**2 - 3) * _SKP[("d", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * l * (l**2 - m**2 - 1) * (5 * l**2 - 1) * _SKP[("d", "f", "P")]
            - sp.sqrt(15) / 4 * l * ((l**2 - m**2) * (1 - l**2) - 2 * n**2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dv", "fay"):
        return (
            sp.sqrt(3) / 4 * m * (l**2 - m**2) * (5 * m**2 - 3) * _SKP[("d", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * m * (l**2 - m**2 + 1) * (5 * m**2 - 1) * _SKP[("d", "f", "P")]
            - sp.sqrt(15) / 4 * m * ((l**2 - m**2) * (1 - m**2) + 2 * n**2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dv", "faz"):
        return (
            sp.sqrt(3) / 4 * n * (l**2 - m**2) * (5 * n**2 - 3) * _SKP[("d", "f", "S")]
            - sp.sqrt(3) / sp.sqrt(8) * n * (l**2 - m**2) * (5 * n**2 - 1) * _SKP[("d", "f", "P")]
            + sp.sqrt(15) / 4 * n * (n**2 + 1) * (l**2 - m**2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dv", "fbx"):
        return (
            sp.S(3) * sp.sqrt(5) / 4 * l * (l**2 - m**2) * (m**2 - n**2) * _SKP[("d", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * l * (3 * (l**2 - m**2) * (m**2 - n**2) - l**2 + 1) * _SKP[("d", "f", "P")]
            + sp.S(1) / 4 * l * (3 * (l**2 - m**2) * (m**2 - n**2) - 4 * l**2 + 2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dv", "fby"):
        return (
            sp.S(3) * sp.sqrt(5) / 4 * m * (l**2 - m**2) * (n**2 - l**2) * _SKP[("d", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * m * (3 * (l**2 - m**2) * (n**2 - l**2) - m**2 + 1) * _SKP[("d", "f", "P")]
            + sp.S(1) / 4 * m * (3 * (l**2 - m**2) * (n**2 - l**2) - 4 * m**2 + 2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("dv", "fbz"):
        return (
            sp.S(3) * sp.sqrt(5) / 4 * n * (l**2 - m**2) ** 2 * _SKP[("d", "f", "S")]
            - sp.sqrt(5) / sp.sqrt(8) * n * (3 * (l**2 - m**2) ** 2 + 2 * n**2 - 2) * _SKP[("d", "f", "P")]
            + sp.S(1) / 4 * n * (3 * (l**2 - m**2) ** 2 + 8 * n**2 - 4) * _SKP[("d", "f", "D")]
        )
    ###
    elif (o1, o2) == ("du", "fxyz"):
        return (
            sp.sqrt(15) / 2 * l * m * n * (3 * n**2 - 1) * _SKP[("d", "f", "S")]
            - sp.sqrt(15) / sp.sqrt(2) * l * m * n * (3 * n**2 - 1) * _SKP[("d", "f", "P")]
            + sp.sqrt(3) / 2 * l * m * n * (3 * n**2 - 1) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("du", "fax"):
        return (
            sp.S(1) / 4 * l * (3 * n**2 - 1) * (5 * l**2 - 3) * _SKP[("d", "f", "S")]
            - sp.S(3) / sp.sqrt(2) / 4 * l * n**2 * (5 * l**2 - 1) * _SKP[("d", "f", "P")]
            + sp.S(3) * sp.sqrt(5) / 4 * l * (l**2 * n**2 - m**2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("du", "fay"):
        return (
            sp.S(1) / 4 * m * (3 * n**2 - 1) * (5 * m**2 - 3) * _SKP[("d", "f", "S")]
            - sp.S(3) / sp.sqrt(2) / 4 * m * n**2 * (5 * m**2 - 1) * _SKP[("d", "f", "P")]
            + sp.S(3) * sp.sqrt(5) / 4 * m * (m**2 * n**2 - l**2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("du", "faz"):
        return (
            sp.S(1) / 4 * n * (3 * n**2 - 1) * (5 * n**2 - 3) * _SKP[("d", "f", "S")]
            - sp.S(3) / sp.sqrt(2) / 4 * n * (5 * n**2 - 1) * (n**2 - 1) * _SKP[("d", "f", "P")]
            + sp.S(3) * sp.sqrt(5) / 4 * n * (n**2 - 1) ** 2 * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("du", "fbx"):
        return (
            sp.sqrt(15) / 4 * l * (m**2 - n**2) * (3 * n**2 - 1) * _SKP[("d", "f", "S")]
            - sp.sqrt(15) / sp.sqrt(8) * l * n**2 * (3 * (m**2 - n**2) + 2) * _SKP[("d", "f", "P")]
            + sp.sqrt(3) / 4 * l * ((3 * n**2 - 1) * (m**2 - n**2) - 4 * l**2 + 2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("du", "fby"):
        return (
            sp.sqrt(15) / 4 * m * (n**2 - l**2) * (3 * n**2 - 1) * _SKP[("d", "f", "S")]
            - sp.sqrt(15) / sp.sqrt(8) * m * n**2 * (3 * (n**2 - l**2) - 2) * _SKP[("d", "f", "P")]
            + sp.sqrt(3) / 4 * m * ((3 * n**2 - 1) * (n**2 - l**2) + 4 * m**2 - 2) * _SKP[("d", "f", "D")]
        )
    elif (o1, o2) == ("du", "fbz"):
        return (
            sp.sqrt(15) / 4 * n * (3 * n**2 - 1) * (l**2 - m**2) * _SKP[("d", "f", "S")]
            - sp.sqrt(15) / sp.sqrt(8) * n * (3 * n**2 - 1) * (l**2 - m**2) * _SKP[("d", "f", "P")]
            + sp.sqrt(3) / 4 * n * (3 * n**2 - 1) * (l**2 - m**2) * _SKP[("d", "f", "D")]
        )
    ###
    elif (o1, o2) == ("fxyz", "fxyz"):
        return (
            15 * l**2 * m**2 * n**2 * _SKP[("f", "f", "S")]
            + sp.S(5) / 2 * (l**2 * m**2 + m**2 * n**2 + n**2 * l**2 - 9 * l**2 * m**2 * n**2) * _SKP[("f", "f", "P")]
            + (1 - 4 * (l**2 * m**2 + m**2 * n**2 + n**2 * l**2) + 9 * l**2 * m**2 * n**2) * _SKP[("f", "f", "D")]
            + sp.S(3) / 2 * (1 - l**2) * (1 - m**2) * (1 - n**2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fxyz", "fax"):
        return (
            sp.sqrt(15) / 2 * m * n * l**2 * (5 * l**2 - 3) * _SKP[("f", "f", "S")]
            - sp.sqrt(15) / 4 * m * n * (3 * l**2 - 1) * (5 * l**2 - 1) * _SKP[("f", "f", "P")]
            + sp.sqrt(15) / 2 * m * n * l**2 * (3 * l**2 - 1) * _SKP[("f", "f", "D")]
            + sp.sqrt(15) / 4 * m * n * (1 - l**4) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fxyz", "fay"):
        return (
            sp.sqrt(15) / 2 * n * l * m**2 * (5 * m**2 - 3) * _SKP[("f", "f", "S")]
            - sp.sqrt(15) / 4 * n * l * (3 * m**2 - 1) * (5 * m**2 - 1) * _SKP[("f", "f", "P")]
            + sp.sqrt(15) / 2 * n * l * m**2 * (3 * m**2 - 1) * _SKP[("f", "f", "D")]
            + sp.sqrt(15) / 4 * n * l * (1 - m**4) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fxyz", "faz"):
        return (
            sp.sqrt(15) / 2 * l * m * n**2 * (5 * n**2 - 3) * _SKP[("f", "f", "S")]
            - sp.sqrt(15) / 4 * l * m * (3 * n**2 - 1) * (5 * n**2 - 1) * _SKP[("f", "f", "P")]
            + sp.sqrt(15) / 2 * l * m * n**2 * (3 * n**2 - 1) * _SKP[("f", "f", "D")]
            + sp.sqrt(15) / 4 * l * m * (1 - n**4) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fxyz", "fbx"):
        return (
            sp.S(15) / 2 * m * n * l**2 * (m**2 - n**2) * _SKP[("f", "f", "S")]
            - sp.S(5) / 4 * m * n * (m**2 - n**2) * (9 * l**2 - 1) * _SKP[("f", "f", "P")]
            + sp.S(1) / 2 * m * n * (m**2 - n**2) * (9 * l**2 - 4) * _SKP[("f", "f", "D")]
            + sp.S(3) / 4 * m * n * (m**2 - n**2) * (1 - l**2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fxyz", "fby"):
        return (
            sp.S(15) / 2 * n * l * m**2 * (n**2 - l**2) * _SKP[("f", "f", "S")]
            - sp.S(5) / 4 * n * l * (n**2 - l**2) * (9 * m**2 - 1) * _SKP[("f", "f", "P")]
            + sp.S(1) / 2 * n * l * (n**2 - l**2) * (9 * m**2 - 4) * _SKP[("f", "f", "D")]
            + sp.S(3) / 4 * n * l * (n**2 - l**2) * (1 - m**2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fxyz", "fbz"):
        return (
            sp.S(15) / 2 * l * m * n**2 * (l**2 - m**2) * _SKP[("f", "f", "S")]
            - sp.S(5) / 4 * l * m * (l**2 - m**2) * (9 * n**2 - 1) * _SKP[("f", "f", "P")]
            + sp.S(1) / 2 * l * m * (l**2 - m**2) * (9 * n**2 - 4) * _SKP[("f", "f", "D")]
            + sp.S(3) / 4 * l * m * (l**2 - m**2) * (1 - n**2) * _SKP[("f", "f", "Phi")]
        )
    ###
    elif (o1, o2) == ("fax", "fax"):
        return (
            sp.S(1) / 4 * l**2 * (5 * l**2 - 3) ** 2 * _SKP[("f", "f", "S")]
            + sp.S(3) / 8 * (5 * l**2 - 1) ** 2 * (1 - l**2) * _SKP[("f", "f", "P")]
            + sp.S(15) / 4 * l**2 * (1 - l**2) ** 2 * _SKP[("f", "f", "D")]
            + sp.S(5) / 8 * (1 - l**2) ** 3 * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fax", "fay"):
        return (
            sp.S(1) / 4 * m * l * (5 * m**2 - 3) * (5 * l**2 - 3) * _SKP[("f", "f", "S")]
            - sp.S(3) / 8 * m * l * (5 * m**2 - 1) * (5 * l**2 - 1) * _SKP[("f", "f", "P")]
            + sp.S(15) / 4 * m * l * (m**2 * l**2 - n**2) * _SKP[("f", "f", "D")]
            + sp.S(5) / 8 * m * l * (3 * n**2 - m**2 * l**2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fax", "faz"):
        return (
            sp.S(1) / 4 * n * l * (5 * n**2 - 3) * (5 * l**2 - 3) * _SKP[("f", "f", "S")]
            - sp.S(3) / 8 * n * l * (5 * n**2 - 1) * (5 * l**2 - 1) * _SKP[("f", "f", "P")]
            + sp.S(15) / 4 * n * l * (n**2 * l**2 - m**2) * _SKP[("f", "f", "D")]
            + sp.S(5) / 8 * n * l * (3 * m**2 - n**2 * l**2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fax", "fbx"):
        return (
            sp.sqrt(15) / 4 * (m**2 - n**2) * l**2 * (5 * l**2 - 3) * _SKP[("f", "f", "S")]
            - sp.sqrt(15) / 8 * (m**2 - n**2) * (5 * l**2 - 1) * (3 * l**2 - 1) * _SKP[("f", "f", "P")]
            + sp.sqrt(15) / 4 * (m**2 - n**2) * l**2 * (3 * l**2 - 1) * _SKP[("f", "f", "D")]
            + sp.sqrt(15) / 8 * (m**2 - n**2) * (1 - l**4) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fax", "fby"):
        return (
            sp.sqrt(15) / 4 * l * m * (n**2 - l**2) * (5 * l**2 - 3) * _SKP[("f", "f", "S")]
            - sp.sqrt(15) / 8 * l * m * (2 + 3 * (n**2 - l**2)) * (5 * l**2 - 1) * _SKP[("f", "f", "P")]
            + sp.sqrt(15) / 4 * l * m * (3 * (1 + l**2) * (n**2 - l**2) + 8 * l**2 - 2) * _SKP[("f", "f", "D")]
            - sp.sqrt(15) / 8 * l * m * ((l**2 + 3) * (n**2 - l**2) + 6 * l**2 - 2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fax", "fbz"):
        return (
            sp.sqrt(15) / 4 * l * n * (l**2 - m**2) * (5 * l**2 - 3) * _SKP[("f", "f", "S")]
            + sp.sqrt(15) / 8 * l * n * (2 - 3 * (l**2 - m**2)) * (5 * l**2 - 1) * _SKP[("f", "f", "P")]
            + sp.sqrt(15) / 4 * l * n * (3 * (1 + l**2) * (l**2 - m**2) - 8 * l**2 + 2) * _SKP[("f", "f", "D")]
            + sp.sqrt(15) / 8 * l * n * (-(l**2 + 3) * (l**2 - m**2) + 6 * l**2 - 2) * _SKP[("f", "f", "Phi")]
        )
    ###
    elif (o1, o2) == ("fay", "fay"):
        return (
            sp.S(1) / 4 * m**2 * (5 * m**2 - 3) ** 2 * _SKP[("f", "f", "S")]
            + sp.S(3) / 8 * (5 * m**2 - 1) ** 2 * (1 - m**2) * _SKP[("f", "f", "P")]
            + sp.S(15) / 4 * m**2 * (1 - m**2) ** 2 * _SKP[("f", "f", "D")]
            + sp.S(5) / 8 * (1 - m**2) ** 3 * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fay", "faz"):
        return (
            sp.S(1) / 4 * n * m * (5 * n**2 - 3) * (5 * m**2 - 3) * _SKP[("f", "f", "S")]
            - sp.S(3) / 8 * n * m * (5 * n**2 - 1) * (5 * m**2 - 1) * _SKP[("f", "f", "P")]
            + sp.S(15) / 4 * n * m * (n**2 * m**2 - l**2) * _SKP[("f", "f", "D")]
            + sp.S(5) / 8 * n * m * (3 * l**2 - n**2 * m**2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fay", "fbx"):
        return (
            sp.sqrt(15) / 4 * m * l * (m**2 - n**2) * (5 * m**2 - 3) * _SKP[("f", "f", "S")]
            + sp.sqrt(15) / 8 * m * l * (2 - 3 * (m**2 - n**2)) * (5 * m**2 - 1) * _SKP[("f", "f", "P")]
            + sp.sqrt(15) / 4 * m * l * (3 * (1 + m**2) * (m**2 - n**2) - 8 * m**2 + 2) * _SKP[("f", "f", "D")]
            + sp.sqrt(15) / 8 * m * l * (-(m**2 + 3) * (m**2 - n**2) + 6 * m**2 - 2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fay", "fby"):
        return (
            sp.sqrt(15) / 4 * (n**2 - l**2) * m**2 * (5 * m**2 - 3) * _SKP[("f", "f", "S")]
            - sp.sqrt(15) / 8 * (n**2 - l**2) * (5 * m**2 - 1) * (3 * m**2 - 1) * _SKP[("f", "f", "P")]
            + sp.sqrt(15) / 4 * (n**2 - l**2) * m**2 * (3 * m**2 - 1) * _SKP[("f", "f", "D")]
            + sp.sqrt(15) / 8 * (n**2 - l**2) * (1 - m**4) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fay", "fbz"):
        return (
            sp.sqrt(15) / 4 * m * n * (l**2 - m**2) * (5 * m**2 - 3) * _SKP[("f", "f", "S")]
            - sp.sqrt(15) / 8 * m * n * (2 + 3 * (l**2 - m**2)) * (5 * m**2 - 1) * _SKP[("f", "f", "P")]
            + sp.sqrt(15) / 4 * m * n * (3 * (1 + m**2) * (l**2 - m**2) + 8 * m**2 - 2) * _SKP[("f", "f", "D")]
            - sp.sqrt(15) / 8 * m * n * ((m**2 + 3) * (l**2 - m**2) + 6 * m**2 - 2) * _SKP[("f", "f", "Phi")]
        )
    ###
    elif (o1, o2) == ("faz", "faz"):
        return (
            sp.S(1) / 4 * n**2 * (5 * n**2 - 3) ** 2 * _SKP[("f", "f", "S")]
            + sp.S(3) / 8 * (5 * n**2 - 1) ** 2 * (1 - n**2) * _SKP[("f", "f", "P")]
            + sp.S(15) / 4 * n**2 * (1 - n**2) ** 2 * _SKP[("f", "f", "D")]
            + sp.S(5) / 8 * (1 - n**2) ** 3 * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("faz", "fbx"):
        return (
            sp.sqrt(15) / 4 * n * l * (m**2 - n**2) * (5 * n**2 - 3) * _SKP[("f", "f", "S")]
            - sp.sqrt(15) / 8 * n * l * (2 + 3 * (m**2 - n**2)) * (5 * n**2 - 1) * _SKP[("f", "f", "P")]
            + sp.sqrt(15) / 4 * n * l * (3 * (1 + n**2) * (m**2 - n**2) + 8 * n**2 - 2) * _SKP[("f", "f", "D")]
            - sp.sqrt(15) / 8 * n * l * ((n**2 + 3) * (m**2 - n**2) + 6 * n**2 - 2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("faz", "fby"):
        return (
            sp.sqrt(15) / 4 * n * m * (n**2 - l**2) * (5 * n**2 - 3) * _SKP[("f", "f", "S")]
            + sp.sqrt(15) / 8 * n * m * (2 - 3 * (n**2 - l**2)) * (5 * n**2 - 1) * _SKP[("f", "f", "P")]
            + sp.sqrt(15) / 4 * n * m * (3 * (1 + n**2) * (n**2 - l**2) - 8 * n**2 + 2) * _SKP[("f", "f", "D")]
            + sp.sqrt(15) / 8 * n * m * (-(n**2 + 3) * (n**2 - l**2) + 6 * n**2 - 2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("faz", "fbz"):
        return (
            sp.sqrt(15) / 4 * (l**2 - m**2) * n**2 * (5 * n**2 - 3) * _SKP[("f", "f", "S")]
            - sp.sqrt(15) / 8 * (l**2 - m**2) * (5 * n**2 - 1) * (3 * n**2 - 1) * _SKP[("f", "f", "P")]
            + sp.sqrt(15) / 4 * (l**2 - m**2) * n**2 * (3 * n**2 - 1) * _SKP[("f", "f", "D")]
            + sp.sqrt(15) / 8 * (l**2 - m**2) * (1 - n**4) * _SKP[("f", "f", "Phi")]
        )
    ###
    elif (o1, o2) == ("fbx", "fbx"):
        return (
            sp.S(15) / 4 * l**2 * (m**2 - n**2) ** 2 * _SKP[("f", "f", "S")]
            + sp.S(5) / 8 * (4 * l**2 * (1 - l**2) + (m**2 - n**2) ** 2 * (1 - 9 * l**2)) * _SKP[("f", "f", "P")]
            + sp.S(1) / 4 * ((m**2 - n**2) ** 2 * (9 * l**2 - 4) + 4 * (1 - 2 * l**2) ** 2) * _SKP[("f", "f", "D")]
            + sp.S(3) / 8 * (1 - l**2) * ((1 + l**2) ** 2 - 4 * m**2 * n**2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fbx", "fby"):
        return (
            sp.S(15) / 4 * m * l * (m**2 - n**2) * (n**2 - l**2) * _SKP[("f", "f", "S")]
            - sp.S(5) / 8 * m * l * (9 * (m**2 - n**2) * (n**2 - l**2) - 2 * n**2 + 2) * _SKP[("f", "f", "P")]
            + sp.S(1) / 4 * m * l * (9 * (m**2 - n**2) * (n**2 - l**2) - 8 * n**2 + 2) * _SKP[("f", "f", "D")]
            + sp.S(3) / 8 * m * l * (-(m**2 - n**2) * (n**2 - l**2) + 2 * n**2 + 2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fbx", "fbz"):
        return (
            sp.S(15) / 4 * l * n * (l**2 - m**2) * (m**2 - n**2) * _SKP[("f", "f", "S")]
            - sp.S(5) / 8 * l * n * (9 * (l**2 - m**2) * (m**2 - n**2) - 2 * m**2 + 2) * _SKP[("f", "f", "P")]
            + sp.S(1) / 4 * l * n * (9 * (l**2 - m**2) * (m**2 - n**2) - 8 * m**2 + 2) * _SKP[("f", "f", "D")]
            + sp.S(3) / 8 * l * n * (-(l**2 - m**2) * (m**2 - n**2) + 2 * m**2 + 2) * _SKP[("f", "f", "Phi")]
        )
    ###
    elif (o1, o2) == ("fby", "fby"):
        return (
            sp.S(15) / 4 * m**2 * (n**2 - l**2) ** 2 * _SKP[("f", "f", "S")]
            + sp.S(5) / 8 * (4 * m**2 * (1 - m**2) + (n**2 - l**2) ** 2 * (1 - 9 * m**2)) * _SKP[("f", "f", "P")]
            + sp.S(1) / 4 * ((n**2 - l**2) ** 2 * (9 * m**2 - 4) + 4 * (1 - 2 * m**2) ** 2) * _SKP[("f", "f", "D")]
            + sp.S(3) / 8 * (1 - m**2) * ((1 + m**2) ** 2 - 4 * n**2 * l**2) * _SKP[("f", "f", "Phi")]
        )
    elif (o1, o2) == ("fby", "fbz"):
        return (
            sp.S(15) / 4 * m * n * (l**2 - m**2) * (n**2 - l**2) * _SKP[("f", "f", "S")]
            - sp.S(5) / 8 * m * n * (9 * (l**2 - m**2) * (n**2 - l**2) - 2 * l**2 + 2) * _SKP[("f", "f", "P")]
            + sp.S(1) / 4 * m * n * (9 * (l**2 - m**2) * (n**2 - l**2) - 8 * l**2 + 2) * _SKP[("f", "f", "D")]
            + sp.S(3) / 8 * m * n * (-(l**2 - m**2) * (n**2 - l**2) + 2 * l**2 + 2) * _SKP[("f", "f", "Phi")]
        )
    ###
    elif (o1, o2) == ("fbz", "fbz"):
        return (
            sp.S(15) / 4 * n**2 * (l**2 - m**2) ** 2 * _SKP[("f", "f", "S")]
            + sp.S(5) / 8 * (4 * n**2 * (1 - n**2) + (l**2 - m**2) ** 2 * (1 - 9 * n**2)) * _SKP[("f", "f", "P")]
            + sp.S(1) / 4 * ((l**2 - m**2) ** 2 * (9 * n**2 - 4) + 4 * (1 - 2 * n**2) ** 2) * _SKP[("f", "f", "D")]
            + sp.S(3) / 8 * (1 - n**2) * ((1 + n**2) ** 2 - 4 * l**2 * m**2) * _SKP[("f", "f", "Phi")]
        )
