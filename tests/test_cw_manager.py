"""
tests for CWManager.
"""

from symclosestwannier.cw.cw_manager import CWManager


# ==================================================
def test_formatter_is_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert CWManager(formatter=False).formatter is False
    assert CWManager(formatter=True).formatter is True
