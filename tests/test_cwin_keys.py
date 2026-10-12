"""
input keywords of seedname.cwin must be documented consistently.

CWin._default() defines the accepted keywords; they are also listed in
util/header.py (shown by `pw2cw -i`), the CWin.read docstring and docs/src/format/cwin.md.
"""

import ast
import os
import re

import pytest

from symclosestwannier.cw.cwin import CWin
from symclosestwannier.util.header import cwin_info

from conftest import TOPDIR

KEYS = set(CWin._default().keys())


# ==================================================
def listed_keys(text):
    return set(re.findall(r"^\s*- (\w+)\s*:", text, re.M))


# ==================================================
@pytest.mark.parametrize(
    "name, keys",
    [
        ("util/header.py (pw2cw -i)", set(cwin_info.keys())),
        ("CWin.read docstring", listed_keys(CWin.read.__doc__)),
        ("docs/src/format/cwin.md", listed_keys(open(os.path.join(TOPDIR, "docs", "src", "format", "cwin.md")).read())),
    ],
)
def test_keys_are_documented(name, keys):
    assert sorted(KEYS - keys) == [], f"keywords missing in {name}"
    assert sorted(keys - KEYS) == [], f"unknown keywords in {name}"


# ==================================================
def test_header_has_no_duplicated_keys():
    with open(os.path.join(TOPDIR, "symclosestwannier", "util", "header.py")) as fp:
        text = fp.read().split("\nwin_info = {")[0]

    keys = re.findall(r'^\s*"(\w+)":', text, re.M)
    duplicated = sorted({k for k in keys if keys.count(k) > 1})

    assert duplicated == []


NO_DEFAULT = object()


# ==================================================
def documented_default(text):
    """
    default value written as the last [...] of a description, e.g. "... (float), [0.0].".
    """
    text = text.strip().rstrip(".").rstrip()
    if not text.endswith("]"):
        return NO_DEFAULT

    depth = 0
    for i in range(len(text) - 1, -1, -1):
        depth += {"]": 1, "[": -1}.get(text[i], 0)
        if depth == 0:
            s = text[i + 1 : -1].strip().strip("'\"")
            break

    for candidate in (s, s.replace(" ", ",")):
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            pass
    return s


# ==================================================
def described(text):
    return {k: v for k, v in re.findall(r"^\s*- (\w+)\s*:\s*(.*)$", text, re.M)}


# ==================================================
@pytest.mark.parametrize(
    "name, descriptions",
    [
        ("util/header.py (pw2cw -i)", cwin_info),
        ("CWin.read docstring", described(CWin.read.__doc__)),
        ("docs/src/format/cwin.md", described(open(os.path.join(TOPDIR, "docs", "src", "format", "cwin.md")).read())),
    ],
)
def test_documented_defaults(name, descriptions):
    wrong = {}
    for key, value in CWin._default().items():
        doc = documented_default(descriptions[key])
        if doc is NO_DEFAULT:
            continue
        if isinstance(value, list) and not isinstance(doc, list):
            doc = [doc] if not isinstance(doc, tuple) else list(doc)
        if doc != value and not (value == "" and doc in ("", " ")):
            wrong[key] = (value, doc)

    assert wrong == {}, f"documented defaults differ from CWin._default() in {name}: {{key: (actual, documented)}}"


# ==================================================
def test_documented_none_is_compared():
    assert documented_default("lattice parameter (float), [None].") is None
    assert documented_default("number of electrons per unit-cell") is NO_DEFAULT


# ==================================================
@pytest.mark.parametrize(
    "line, key, value",
    [
        ("dos_kmesh = 2 3 4", "dos_kmesh", [2, 3, 4]),
        ("cohp_kmesh : 2 3 4", "cohp_kmesh", [2, 3, 4]),
        ("lindhard_kmesh =\t2\t3  4 ! comment", "lindhard_kmesh", [2, 3, 4]),
        ("fermi_surface_kmesh = -1 1 10 -2 2 20", "fermi_surface_kmesh", [[-1, 1, 10], [-2, 2, 20]]),
        ("fermi_surface_view : 0 1 0", "fermi_surface_view", [0, 1, 0]),
        ("lindhard_surface_qmesh = -1 1 10   -2 2 20 ! comment", "lindhard_surface_qmesh", [[-1, 1, 10], [-2, 2, 20]]),
        ("lindhard_surface_view = 0 0 1", "lindhard_surface_view", [0, 0, 1]),
    ],
)
def test_mesh_parsing(tmp_path, line, key, value):
    (tmp_path / "seed.cwin").write_text(line + "\n")

    assert CWin(str(tmp_path), "seed")[key] == value


# ==================================================
def test_qpoint_path_parsing(tmp_path):
    (tmp_path / "seed.cwin").write_text("""
begin qpoint_path
G 0 0 0 X 0.5 0 0
X 0.5 0 0 M 0.5 0.5 0
G 0 0 0 M 0.5 0.5 0
end qpoint_path
""")

    cwin = CWin(str(tmp_path), "seed")

    assert cwin["qpoint_path"] == "G-X-M|G-M"
    assert cwin["qpoint"] == {"G": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0], "M": [0.5, 0.5, 0.0]}
