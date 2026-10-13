"""
tests for CWManager.
"""

from symclosestwannier.cw.cw_manager import CWManager


# ==================================================
def test_formatter_is_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert CWManager(formatter=False).formatter is False
    assert CWManager(formatter=True).formatter is True


# ==================================================
def test_write_to_absolute_and_relative_paths(tmp_path, monkeypatch):
    # mp_outdir is an absolute path, joined to file names by cw_creator.
    work = tmp_path / "work"
    other = tmp_path / "model"
    work.mkdir()
    other.mkdir()
    monkeypatch.chdir(work)
    cwm = CWManager(formatter=False)

    cwm.write(str(other / "model_z.dat.cw"), {"z1": 1.0})
    cwm.write("seed_hr.dat.cw", {"z1": 1.0})

    assert (other / "model_z.dat.cw").is_file()
    assert (work / "seed_hr.dat.cw").is_file()
    assert cwm.filename(str(other / "x")) == str(other / "x")
