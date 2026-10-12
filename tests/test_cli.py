"""
tests for pw2cw/postcw command line interface and error messages.
"""

import pickle

import pytest
from click.testing import CliRunner

from symclosestwannier.__init__ import __version__
from symclosestwannier.scripts.pw2cw import cmd as pw2cw
from symclosestwannier.scripts.postcw import cmd as postcw
from symclosestwannier.util.exceptions import SymCWError, SymCWFileNotFoundError, SymCWInputError

COMMANDS = [pw2cw, postcw]


# ==================================================
@pytest.mark.parametrize("command", COMMANDS)
def test_version(command, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(command, ["-v", "seed"])

    assert result.exit_code == 0
    assert result.output.strip() == f"SymClosestWannier: {__version__}"


# ==================================================
@pytest.mark.parametrize("command", COMMANDS)
def test_input_format(command):
    result = CliRunner().invoke(command, ["-i"])

    assert result.exit_code == 0
    assert "outdir" in result.output


# ==================================================
@pytest.mark.parametrize("command", COMMANDS)
def test_missing_seedname(command):
    result = CliRunner().invoke(command, [])

    assert result.exit_code == 2
    assert "missing SEEDNAME" in result.output


# ==================================================
@pytest.mark.parametrize("command", COMMANDS)
def test_missing_cwin(command, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(command, ["seed.cwin"])

    assert result.exit_code == 1
    assert "Error: cannot find the cwin file:" in result.output
    assert "seed.cwin" in result.output
    assert "Traceback" not in result.output


# ==================================================
@pytest.mark.parametrize(
    "extra, message",
    [
        ("delta = 1e-12", "invalid keyword = delta was given. did you mean 'cwf_delta'"),
        ("ket_amn = [pz@C_1,pz@C_2]", "invalid ket_amn = [pz@C_1,pz@C_2] was given. each ket must be"),
        ("proj_min = abc", "invalid value proj_min = abc"),
        ("restart = foo", "invalid restart = foo was given."),
        ("disentangle = true", "cwf_mu_max and cwf_mu_min must be specified when disentangle = true."),
        ("disentangle = true\ncwf_mu_min = 1.0\ncwf_mu_max = 0.0", "check disentanglement windows"),
        ("dos_kmesh = 1 a 1", "invalid value dos_kmesh = 1 a 1"),
        ("fermi_surface_kmesh = -1 1 10", "fermi_surface_kmesh needs 6 integers"),
        ("begin qpoint_path\nG 0 0 0 X 0.5 0\nend qpoint_path", "invalid qpoint_path block"),
        ("begin qpoint_path\nG 0 0 0 X 0.5 0 0", "invalid qpoint_path block"),
        ("begin qpoint_path\nG 0 0 0 X 0.5 a 0\nend qpoint_path", "invalid qpoint_path block"),
        ("lindhard_surface_qmesh = -1 1 10 -1 1", "lindhard_surface_qmesh needs 6 integers"),
        ("fermi_energy = 0.0", "invalid keyword = fermi_energy was given. fermi_energy must be given in seedname.win."),
    ],
)
def test_invalid_cwin(make_case, extra, message):
    workdir = make_case("ch4_sl", extra=extra)

    result = CliRunner().invoke(pw2cw, ["ch4_sl"])

    assert result.exit_code == 1
    assert f"Error: {workdir / 'ch4_sl.cwin'}: " in result.output
    assert message in result.output


# ==================================================
@pytest.mark.parametrize("command", COMMANDS)
def test_multiple_seednames(command):
    result = CliRunner().invoke(command, ["a", "b"])

    assert result.exit_code == 2
    assert "unexpected extra argument" in result.output


# ==================================================
def test_proj_min_too_large(make_case):
    make_case("ch4_sl", extra="proj_min = 1.1")

    result = CliRunner().invoke(pw2cw, ["ch4_sl"])

    assert result.exit_code == 1
    assert "proj_min = 1.1 is too large" in result.output
    assert "num_wann = 8" in result.output


# ==================================================
def test_postcw_without_mmn(make_case, monkeypatch):
    workdir = make_case("ch4_sl", extra="write_info_data = true")
    assert CliRunner().invoke(pw2cw, ["ch4_sl"]).exit_code == 0

    monkeypatch.chdir(workdir)
    result = CliRunner().invoke(postcw, ["ch4_sl"])

    assert result.exit_code == 1
    assert "requires ch4_sl.mmn" in result.output


# ==================================================
@pytest.mark.parametrize(
    "error",
    [SymCWInputError("invalid"), SymCWFileNotFoundError("eig", "a.eig"), SymCWFileNotFoundError("eig", "a.eig", "hint")],
)
def test_exceptions_can_be_pickled(error):
    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is type(error)
    assert str(restored) == str(error)
    if isinstance(error, SymCWFileNotFoundError):
        assert (restored.kind, restored.file_name, restored.hint) == (error.kind, error.file_name, error.hint)


# ==================================================
def test_exceptions_derive_from_builtin():
    assert issubclass(SymCWInputError, ValueError)
    assert issubclass(SymCWFileNotFoundError, FileNotFoundError)
    assert issubclass(SymCWFileNotFoundError, OSError)
    assert issubclass(SymCWFileNotFoundError, SymCWError)
    assert str(SymCWFileNotFoundError("eig", "a.eig")) == "cannot find the eig file: a.eig"


# ==================================================
def test_handled_errors_are_not_raised(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(pw2cw, ["seed"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "Error: cannot find the cwin file:" in result.output


# ==================================================
def test_unexpected_errors_are_raised(monkeypatch):
    def fail(seedname):
        raise RuntimeError("unexpected")

    monkeypatch.setattr("symclosestwannier.scripts.pw2cw.cw_creator", fail)

    with pytest.raises(RuntimeError, match="unexpected"):
        CliRunner().invoke(pw2cw, ["seed"], catch_exceptions=False)


# ==================================================
def test_cw_manager_read_missing_file(tmp_path):
    from symclosestwannier.cw.cw_manager import CWManager

    cwm = CWManager(topdir=str(tmp_path), verbose=False, parallel=False, formatter=False)

    with pytest.raises(SymCWFileNotFoundError, match="cannot find the dict file:"):
        cwm.read("missing.dat")
