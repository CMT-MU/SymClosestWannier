"""
outdir and mp_outdir are written to seedname.cwout and seedname.hdf5 relative to the directory of seedname.cwin,
so that the outputs do not contain the absolute path of the directory where they were created.
"""

import h5py
import pytest

from symclosestwannier.cw.cw_creator import cw_creator


# ==================================================
@pytest.mark.parametrize("outdir, expected", [("./", "."), ("./out", "out")])
def test_relative_outdir_in_outputs(make_case, outdir, expected):
    seedname = "ch4_sl"
    workdir = make_case(seedname, outdir=outdir, extra="write_info_data = true")

    cw_creator(seedname)

    outpath = workdir / expected
    cwout = (outpath / f"{seedname}.cwout").read_text()
    assert str(workdir) not in cwout
    rows = {
        line.split(":")[0].strip("| ").strip(): line.split(":")[1].strip(" |")
        for line in cwout.splitlines()
        if "outdir" in line and line.startswith("|")
    }
    assert rows["outdir"] == expected
    assert rows["mp_outdir"] == "."

    with h5py.File(outpath / f"{seedname}.hdf5", "r") as hf:
        info = hf["info"]
        assert info["outdir"][()].decode() == expected
        assert info["mp_outdir"][()].decode() == "."
        assert "cwin_dir" not in info
        for k, v in info.items():
            v = v[()]
            if isinstance(v, bytes):
                assert str(workdir) not in v.decode(errors="ignore"), k
