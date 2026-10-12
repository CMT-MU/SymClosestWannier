"""
tests for input/output directory handling of pw2cw and postcw.

inputs (seedname.cwin/win/amn/..., seedname.band.gnu, hr_input, MultiPie model) are read from
the directory where pw2cw/postcw is run, outputs are written to outdir/mp_outdir.
"""

import os
import shutil

import numpy as np
import pytest

import symclosestwannier.analyzer.analyzer as analyzer_module
import symclosestwannier.cw.cw_info as cw_info_module
import symclosestwannier.util.utility as utility_module
from symclosestwannier.cw.cw_creator import cw_creator, _ref_band_filename
from symclosestwannier.util.utility import run_gnuplot

from conftest import read_hr_dat


# ==================================================
@pytest.mark.parametrize("ext", ["band.gnu", "band.gnu.dat"])
def test_ref_band_filename(tmp_path, ext):
    indir = tmp_path / "in"
    plotdir = tmp_path / "out" / "sym"
    indir.mkdir()
    plotdir.mkdir(parents=True)
    (indir / f"seed.{ext}").write_text("")

    assert _ref_band_filename(str(indir), "seed", str(plotdir)) == f"../../in/seed.{ext}"
    assert _ref_band_filename(str(indir), "other", str(plotdir)) is None


# ==================================================
def test_ref_band_filename_prefers_band_gnu(tmp_path):
    for ext in ("band.gnu", "band.gnu.dat"):
        (tmp_path / f"seed.{ext}").write_text("")

    assert _ref_band_filename(str(tmp_path), "seed", str(tmp_path)) == "seed.band.gnu"


# ==================================================
def test_run_gnuplot_without_gnuplot(tmp_path, monkeypatch, capsys):
    def missing(*args, **kwargs):
        raise FileNotFoundError("gnuplot")

    monkeypatch.setattr(utility_module.subprocess, "run", missing)

    run_gnuplot(str(tmp_path), "plot_band.gnu")

    assert "gnuplot is not found" in capsys.readouterr().err


# ==================================================
def test_run_gnuplot_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="directory"):
        run_gnuplot(str(tmp_path / "missing"), "plot_band.gnu")


# ==================================================
def test_band_plot_in_directories_with_spaces(make_case, monkeypatch):
    """
    gnuplot runs in outdir without going through the shell, and the reference DFT band
    file written in the script is found relative to outdir.
    """
    calls = []
    monkeypatch.setattr(utility_module.subprocess, "run", lambda args, cwd: calls.append((args, os.path.abspath(cwd))))

    seedname = "graphene_pz"
    workdir = make_case(seedname, outdir="./out", dirname="work dir", band=True)

    cw_creator(seedname)

    outdir = workdir / "out"
    assert (["gnuplot", "plot_band.gnu"], str(outdir)) in calls

    script = (outdir / "plot_band.gnu").read_text()
    ref = "../graphene_pz.band.gnu"
    assert f"'{ref}'" in script
    assert (outdir / ref).is_file()


# ==================================================
def test_symmetrization_loads_multipie_model_from_input_directory(make_case, monkeypatch):
    class Loaded(Exception):
        pass

    class MaterialModelStub:
        def __init__(self, topdir=None, verbose=False):
            self.topdir = topdir

        def load(self, name):
            raise Loaded(self.topdir, name)

    monkeypatch.setattr(cw_info_module, "MaterialModel", MaterialModelStub)

    seedname = "ch4_sl"
    workdir = make_case(seedname, outdir="./out", extra="symmetrization = true\nmp_seedname = model")

    with pytest.raises(Loaded) as e:
        cw_creator(seedname)

    assert e.value.args == (str(workdir), "model")


# ==================================================
class ResponseStub:
    created = []

    def __init__(self, cwi, cwm, HH_R=None):
        ResponseStub.created.append((cwi, HH_R))


class BandStub:
    def __init__(self, cwi, cwm):
        pass


# ==================================================
@pytest.fixture
def postcw_stubbed(monkeypatch):
    """
    replace response/band calculations of postcw, which need seedname.mmn not included in the examples.
    """
    ResponseStub.created = []
    monkeypatch.setattr(analyzer_module, "Response", ResponseStub)
    monkeypatch.setattr(analyzer_module, "Band", BandStub)
    return ResponseStub.created


# ==================================================
def test_postcw_with_outdir_and_hr_input(make_case, monkeypatch, postcw_stubbed):
    seedname = "ch4_sl"
    workdir = make_case(seedname, outdir="./out", extra="write_hr = true\nwrite_info_data = true")
    cw_creator(seedname)

    # hr_input is relative to the directory where postcw is run.
    shutil.copy(workdir / "out" / f"{seedname}_hr.dat.cw", workdir / "hr_input.dat")
    with open(workdir / f"{seedname}.cwin", "a") as fp:
        fp.write("hr_input = hr_input.dat\n")

    monkeypatch.chdir(workdir)
    analyzer_module.analyzer(seedname)

    assert (workdir / "out" / f"{seedname}.cwpout").is_file()

    cwi, HH_R = postcw_stubbed[0]
    _, _, Hr_ref = read_hr_dat(workdir / "hr_input.dat")
    np.testing.assert_allclose(HH_R, Hr_ref, rtol=0, atol=1e-7)


# ==================================================
def test_postcw_after_moving_directory(make_case, monkeypatch, postcw_stubbed):
    """
    directories stored in seedname.hdf5 by pw2cw are overridden by the current seedname.cwin.
    """
    seedname = "ch4_sl"
    workdir = make_case(seedname, outdir="./out", extra="write_info_data = true", dirname="before")
    cw_creator(seedname)

    moved = workdir.parent / "after"
    monkeypatch.chdir(workdir.parent)
    shutil.move(workdir, moved)

    monkeypatch.chdir(moved)
    analyzer_module.analyzer(seedname)

    cwi, _ = postcw_stubbed[0]
    assert cwi["outdir"] == str(moved / "out")
    assert cwi["mp_outdir"] == str(moved)
    assert os.getcwd() == str(moved / "out")
