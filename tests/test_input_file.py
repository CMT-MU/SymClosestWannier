"""
tests for reading input files compressed as seedname.ext.gz or seedname.ext.tar.gz.
"""

import io
import os
import re
import gzip
import shutil
import tarfile
import zlib

import numpy as np
import pytest
from scipy.io import FortranFile

from symclosestwannier.cw.cw_creator import cw_creator
from symclosestwannier.cw.mmn import Mmn
from symclosestwannier.cw.spn import Spn
from symclosestwannier.cw.uHu import UHu
from symclosestwannier.util.exceptions import SymCWFileNotFoundError, SymCWInputError
from symclosestwannier.util.input_file import open_input, input_path

from conftest import DATA_DIR, EXAMPLES_DIR, assert_hr_equal


# ==================================================
def compress(path, kind, member_dir=""):
    """
    replace path by path.gz or path.tar.gz.

    Args:
        path (pathlib.Path): file to compress.
        kind (str): "gz" or "tar.gz".
        member_dir (str, optional): directory of the member in tar archive.
    """
    if kind == "gz":
        with open(path, "rb") as src, gzip.open(f"{path}.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(f"{path}.tar.gz", "w:gz") as tf:
            tf.add(path, arcname=os.path.join(member_dir, path.name))
    os.remove(path)


# ==================================================
@pytest.mark.parametrize("kind", ["gz", "tar.gz"])
def test_pw2cw_with_compressed_inputs(make_case, kind):
    seedname = "ch4_sl"
    workdir = make_case(seedname, extra="write_hr = true")
    for ext in ("eig", "amn", "nnkp"):
        compress(workdir / f"{seedname}.{ext}", kind, member_dir="data")

    cw_creator(seedname)

    assert_hr_equal(workdir / f"{seedname}_hr.dat.cw", os.path.join(DATA_DIR, f"{seedname}_hr.dat.cw"))


# ==================================================
def test_open_input_prefers_uncompressed(tmp_path):
    (tmp_path / "a.eig").write_text("plain\n")
    with gzip.open(tmp_path / "a.eig.gz", "wt") as fp:
        fp.write("gz\n")

    with open_input(str(tmp_path / "a.eig"), "eig") as fp:
        assert fp.read() == "plain\n"


# ==================================================
def test_open_input_single_member_with_other_name(tmp_path):
    (tmp_path / "other.txt").write_text("data\n")
    with tarfile.open(tmp_path / "a.eig.tar.gz", "w:gz") as tf:
        tf.add(tmp_path / "other.txt", arcname="other.txt")

    with open_input(str(tmp_path / "a.eig"), "eig") as fp:
        assert fp.read() == "data\n"


# ==================================================
def test_open_input_errors(tmp_path):
    with pytest.raises(SymCWFileNotFoundError, match="cannot find the eig file"):
        open_input(str(tmp_path / "a.eig"), "eig")

    for name in ("x.txt", "y.txt"):
        (tmp_path / name).write_text("")
    with tarfile.open(tmp_path / "a.eig.tar.gz", "w:gz") as tf:
        for name in ("x.txt", "y.txt"):
            tf.add(tmp_path / name, arcname=name)

    with pytest.raises(SymCWInputError, match="cannot find a.eig in"):
        open_input(str(tmp_path / "a.eig"), "eig")


# ==================================================
@pytest.mark.parametrize("kind", ["gz", "tar.gz"])
def test_input_path_removes_temporary_file(tmp_path, kind):
    (tmp_path / "a.spn").write_bytes(b"\x00\x01binary")
    compress(tmp_path / "a.spn", kind)

    with input_path(str(tmp_path / "a.spn"), "spn") as path:
        assert path != str(tmp_path / "a.spn")
        assert open(path, "rb").read() == b"\x00\x01binary"

    assert not os.path.exists(path)


# ==================================================
@pytest.mark.parametrize("kind", [None, "gz", "tar.gz"])
def test_mmn(tmp_path, kind):
    (tmp_path / "seed.mmn").write_text("header\n 1 1 1\n 1 1 0 0 0\n 0.5 0.25\n")
    if kind:
        compress(tmp_path / "seed.mmn", kind)

    mmn = Mmn(str(tmp_path), "seed", npar=1)

    np.testing.assert_allclose(mmn["Mkb"], [[[[0.5 + 0.25j]]]])


# ==================================================
def write_unformatted_spn(path):
    with FortranFile(str(path), "w") as f:
        f.write_record(np.frombuffer(b"header", dtype="S1"))
        f.write_record(np.array([1, 1], dtype=np.int32))
        f.write_record(np.array([1.0, 2.0, 3.0], dtype=np.complex128))


# ==================================================
@pytest.mark.parametrize("kind", [None, "gz", "tar.gz"])
@pytest.mark.parametrize("formatted", [True, False])
def test_spn(tmp_path, kind, formatted):
    if formatted:
        (tmp_path / "seed.spn").write_text("header\n 1 1\n 1.0 0.0\n 2.0 0.0\n 3.0 0.0\n")
    else:
        write_unformatted_spn(tmp_path / "seed.spn")
    if kind:
        compress(tmp_path / "seed.spn", kind)

    spn = Spn(str(tmp_path), "seed", formatted=formatted)

    np.testing.assert_allclose(np.array(spn["pauli_spn"]).reshape(3), [1.0, 2.0, 3.0])


# ==================================================
@pytest.mark.parametrize("kind", [None, "gz", "tar.gz"])
def test_uHu_formatted(tmp_path, kind):
    (tmp_path / "seed.uHu").write_text("header\n 1 1 1\n 0.5 -0.25\n")
    if kind:
        compress(tmp_path / "seed.uHu", kind)

    uHu = UHu(str(tmp_path), "seed", formatted=True)

    np.testing.assert_allclose(uHu["Hkb1b2"], [[[[[0.5 - 0.25j]]]]])


# ==================================================
def make_tar(path, members):
    """
    create tar.gz archive with members {name: text}; directories end with "/".
    """
    with tarfile.open(path, "w:gz") as tf:
        for name, text in members.items():
            info = tarfile.TarInfo(name.rstrip("/"))
            if name.endswith("/"):
                info.type = tarfile.DIRTYPE
                tf.addfile(info)
            else:
                data = text.encode()
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))


# ==================================================
def test_open_input_selects_member_by_name(tmp_path):
    make_tar(tmp_path / "a.eig.tar.gz", {"out/": "", "out/a.win": "win\n", "out/a.eig": "eig\n"})

    fp = open_input(str(tmp_path / "a.eig"), "eig")
    with fp:
        assert fp.read() == "eig\n"

    # the tar archive is closed with the file.
    assert fp.buffer._on_close.__self__.closed


# ==================================================
@pytest.mark.parametrize(
    "members",
    [
        {"x/a.eig": "1\n", "y/a.eig": "2\n"},  # ambiguous
        {"a.eig/": ""},  # no regular file
    ],
)
def test_open_input_rejects_archive(tmp_path, members):
    make_tar(tmp_path / "a.eig.tar.gz", members)

    with pytest.raises(SymCWInputError, match="cannot find a.eig in"):
        open_input(str(tmp_path / "a.eig"), "eig")


# ==================================================
def test_open_input_prefers_gz_over_tar(tmp_path):
    with gzip.open(tmp_path / "a.eig.gz", "wt") as fp:
        fp.write("gz\n")
    make_tar(tmp_path / "a.eig.tar.gz", {"a.eig": "tar\n"})

    with open_input(str(tmp_path / "a.eig"), "eig") as fp:
        assert fp.read() == "gz\n"


# ==================================================
@pytest.mark.parametrize("kind", ["gz", "tar.gz"])
@pytest.mark.parametrize("where", ["copy", "body"])
def test_input_path_cleanup_on_error(tmp_path, monkeypatch, kind, where):
    import symclosestwannier.util.input_file as input_file_module

    (tmp_path / "a.spn").write_bytes(b"data")
    compress(tmp_path / "a.spn", kind)

    created = []
    mkstemp = input_file_module.tempfile.mkstemp

    def record_mkstemp(*args, **kwargs):
        fd, path = mkstemp(*args, **kwargs)
        created.append(path)
        return fd, path

    monkeypatch.setattr(input_file_module.tempfile, "mkstemp", record_mkstemp)

    if where == "copy":

        def fail_copy(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(input_file_module.shutil, "copyfileobj", fail_copy)

    with pytest.raises(OSError, match="disk full"):
        with input_path(str(tmp_path / "a.spn"), "spn"):
            raise OSError("disk full")

    assert len(created) == 1
    assert not os.path.exists(created[0])


# ==================================================
def test_input_path_closes_descriptor_when_fdopen_fails(tmp_path, monkeypatch):
    import symclosestwannier.util.input_file as input_file_module

    (tmp_path / "a.spn").write_bytes(b"data")
    compress(tmp_path / "a.spn", "gz")

    closed = []
    monkeypatch.setattr(input_file_module.os, "fdopen", lambda fd, mode: (_ for _ in ()).throw(OSError("fdopen")))
    close = input_file_module.os.close
    monkeypatch.setattr(input_file_module.os, "close", lambda fd: (closed.append(fd), close(fd)))

    with pytest.raises(OSError, match="fdopen"):
        with input_path(str(tmp_path / "a.spn"), "spn"):
            pass

    assert len(closed) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.spn.gz"]


# ==================================================
def test_input_path_priority(tmp_path):
    (tmp_path / "a.spn").write_bytes(b"plain")
    with gzip.open(tmp_path / "a.spn.gz", "wb") as fp:
        fp.write(b"gz")
    make_tar(tmp_path / "a.spn.tar.gz", {"a.spn": "tar"})

    with input_path(str(tmp_path / "a.spn"), "spn") as path:
        assert open(path, "rb").read() == b"plain"

    os.remove(tmp_path / "a.spn")
    with input_path(str(tmp_path / "a.spn"), "spn") as path:
        assert open(path, "rb").read() == b"gz"


# ==================================================
def write_unformatted_uHu(path, num_bands=2, num_k=1, num_b=2):
    rng = np.random.default_rng(0)
    with FortranFile(str(path), "w") as f:
        f.write_record(np.frombuffer(b"header", dtype="S1"))
        f.write_record(np.array([num_bands, num_k, num_b], dtype=np.int32))
        for _ in range(num_k * num_b * num_b):
            f.write_record(rng.standard_normal(2 * num_bands * num_bands))


# ==================================================
@pytest.mark.parametrize("kind", ["gz", "tar.gz"])
def test_uHu_unformatted_compressed_equals_plain(tmp_path, kind):
    plain = tmp_path / "plain"
    packed = tmp_path / "packed"
    for d in (plain, packed):
        d.mkdir()
        write_unformatted_uHu(d / "seed.uHu")
    compress(packed / "seed.uHu", kind)

    expected = np.array(UHu(str(plain), "seed")["Hkb1b2"])
    result = np.array(UHu(str(packed), "seed")["Hkb1b2"])

    assert expected.shape == (1, 2, 2, 2, 2)
    np.testing.assert_array_equal(result, expected)


# ==================================================
def write_umat(path, num_wann=2, num_bands=3, num_k=1):
    rng = np.random.default_rng(1)

    def block(nrow, ncol):
        return "".join(f" {rng.standard_normal():.10f} {rng.standard_normal():.10f}\n" for _ in range(nrow * ncol))

    u = f"header\n {num_k} {num_wann} {num_wann}\n\n"
    u_dis = f"header\n {num_k} {num_wann} {num_bands}\n\n"
    for _ in range(num_k):
        u += " 0.0 0.0 0.0\n" + block(num_wann, num_wann) + "\n"
        u_dis += " 0.0 0.0 0.0\n" + block(num_bands, num_wann) + "\n"
    (path / "ch4_sl_u.mat").write_text(u)
    (path / "ch4_sl_u_dis.mat").write_text(u_dis)


# ==================================================
@pytest.mark.parametrize("kind", ["gz", "tar.gz"])
def test_umat_compressed_equals_plain(tmp_path, kind):
    from symclosestwannier.cw.umat import Umat

    plain = tmp_path / "plain"
    packed = tmp_path / "packed"
    for d in (plain, packed):
        d.mkdir()
        # seedname.win is read for dis_num_iter (= 0 for ch4_sl).
        shutil.copy(os.path.join(EXAMPLES_DIR, "ch4_sl", "ch4_sl.win"), d)
        write_umat(d)
    for name in ("ch4_sl_u.mat", "ch4_sl_u_dis.mat"):
        compress(packed / name, kind)

    expected = Umat(str(plain), "ch4_sl")
    result = Umat(str(packed), "ch4_sl")

    assert np.array(expected["Uk"]).shape == (1, 3, 2)
    for key in ("Uoptk", "Udisk", "Uk"):
        np.testing.assert_array_equal(np.array(result[key]), np.array(expected[key]))


# ==================================================
def break_archive(path, kind, how):
    """
    write a broken compressed file of a text file, which is large enough to be read in several chunks.

    Args:
        path (pathlib.Path): file without .gz or .tar.gz.
        kind (str): "gz" or "tar.gz".
        how (str): "garbage" (not a compressed file), "truncated" or "corrupted" (changed bytes in the middle).

    Returns:
        str: file name of the broken file.
    """
    path.write_text("".join(f"{i} {np.sin(i):.12f}\n" for i in range(200000)))
    compress(path, kind)
    archive = f"{path}.{kind}"

    with open(archive, "rb") as fp:
        data = fp.read()
    if how == "garbage":
        data = b"not a compressed file\n" * 10
    elif how == "truncated":
        data = data[: len(data) // 2]
    else:
        n = len(data) // 2
        data = data[:n] + bytes(b ^ 0xFF for b in data[n : n + 64]) + data[n + 64 :]
    with open(archive, "wb") as fp:
        fp.write(data)

    return archive


# changed bytes in tar.gz are not always detected, since the checksum of gzip is checked only at the end of the archive.
BROKEN_ARCHIVES = [("gz", "garbage"), ("gz", "truncated"), ("gz", "corrupted"), ("tar.gz", "garbage"), ("tar.gz", "truncated")]


# ==================================================
@pytest.mark.parametrize("kind, how", BROKEN_ARCHIVES)
def test_open_input_broken_archive(tmp_path, kind, how):
    archive = break_archive(tmp_path / "a.eig", kind, how)

    with pytest.raises(SymCWInputError, match=re.escape(f"cannot read {archive}, the file may be broken")):
        with open_input(str(tmp_path / "a.eig"), "eig") as fp:
            fp.readlines()


# ==================================================
@pytest.mark.parametrize("kind, how", BROKEN_ARCHIVES)
def test_input_path_broken_archive(tmp_path, monkeypatch, kind, how):
    import symclosestwannier.util.input_file as input_file_module

    archive = break_archive(tmp_path / "a.spn", kind, how)

    created = []
    mkstemp = input_file_module.tempfile.mkstemp

    def record_mkstemp(*args, **kwargs):
        fd, path = mkstemp(*args, **kwargs)
        created.append(path)
        return fd, path

    monkeypatch.setattr(input_file_module.tempfile, "mkstemp", record_mkstemp)

    with pytest.raises(SymCWInputError, match=re.escape(f"cannot read {archive}, the file may be broken")):
        with input_path(str(tmp_path / "a.spn"), "spn"):
            pass

    assert all(not os.path.exists(path) for path in created)


# ==================================================
@pytest.mark.parametrize("kind", ["gz", "tar.gz"])
def test_pw2cw_with_broken_archive(make_case, kind):
    from click.testing import CliRunner

    from symclosestwannier.scripts.pw2cw import cmd as pw2cw

    seedname = "ch4_sl"
    workdir = make_case(seedname)
    compress(workdir / f"{seedname}.eig", kind)
    archive = workdir / f"{seedname}.eig.{kind}"
    archive.write_bytes(archive.read_bytes()[:100])

    result = CliRunner().invoke(pw2cw, [seedname], catch_exceptions=False)

    assert result.exit_code == 1
    assert "Error: cannot read " in result.output
    assert f"{seedname}.eig.{kind}, the file may be broken" in result.output


# ==================================================
@pytest.mark.parametrize("kind", ["gz", "tar.gz"])
def test_open_input_seek_and_name(tmp_path, kind):
    (tmp_path / "a.eig").write_text("1 one\n2 two\n3 three\n")
    compress(tmp_path / "a.eig", kind)

    with open_input(str(tmp_path / "a.eig"), "eig") as fp:
        assert fp.name == f"{tmp_path / 'a.eig'}.{kind}"
        assert fp.seekable()
        assert fp.readline() == "1 one\n"
        pos = fp.tell()
        assert fp.read() == "2 two\n3 three\n"
        fp.seek(pos)
        assert fp.readline() == "2 two\n"
        fp.seek(0)
        assert fp.readline() == "1 one\n"


# ==================================================
class FailingStream(io.BytesIO):
    """
    binary stream which fails as a broken compressed file after the first read, and records close.
    """

    def __init__(self, data, fail_close=False):
        super().__init__(data)
        self.reads = 0
        self.fail_close = fail_close

    def _fail_after_first_read(self):
        self.reads += 1
        if self.reads > 1:
            raise EOFError("Compressed file ended before the end-of-stream marker was reached")

    def read(self, size=-1):
        self._fail_after_first_read()
        return super().read(size)

    def read1(self, size=-1):
        self._fail_after_first_read()
        return super().read1(size)

    def readinto(self, b):
        self._fail_after_first_read()
        return super().readinto(b)

    def seek(self, offset, whence=io.SEEK_SET):
        raise zlib.error("invalid distance too far back")

    def close(self):
        super().close()
        if self.fail_close:
            raise OSError("close failed")


# ==================================================
@pytest.mark.parametrize("method", ["read", "read1", "readinto", "text"])
def test_checked_reader_fails_after_first_read(method):
    from symclosestwannier.util.input_file import _CheckedReader

    raw = FailingStream(b"1\n" * 10)
    closed = []
    src = _CheckedReader(raw, "a.eig.tar.gz", on_close=lambda: closed.append(True))

    with pytest.raises(SymCWInputError, match="cannot read a.eig.tar.gz, the file may be broken") as e:
        if method == "text":
            with io.TextIOWrapper(src) as fp:
                fp.readline()
                fp.read()
        else:
            with src:
                for _ in range(2):
                    if method == "readinto":
                        src.readinto(bytearray(4))
                    else:
                        getattr(src, method)(4)

    assert isinstance(e.value.__cause__, EOFError)
    assert raw.closed and closed == [True] and src.closed


# ==================================================
def test_checked_reader_close_failure():
    from symclosestwannier.util.input_file import _CheckedReader

    raw = FailingStream(b"data", fail_close=True)
    closed = []
    src = _CheckedReader(raw, "a.eig.gz", on_close=lambda: closed.append(True))

    with pytest.raises(OSError, match="close failed"):
        src.close()

    assert closed == [True] and src.closed


# ==================================================
def test_checked_reader_seek_failure():
    from symclosestwannier.util.input_file import _CheckedReader

    with _CheckedReader(FailingStream(b"data"), "a.eig.gz") as src:
        with pytest.raises(SymCWInputError, match=re.escape("cannot read a.eig.gz, the file may be broken")) as e:
            src.seek(0)

    assert isinstance(e.value.__cause__, zlib.error)


# ==================================================
def test_checked_reader_on_close_failure():
    from symclosestwannier.util.input_file import _CheckedReader

    raw = FailingStream(b"data")
    calls = []

    def fail_on_close():
        calls.append(True)
        raise OSError("archive close failed")

    src = _CheckedReader(raw, "a.eig.tar.gz", on_close=fail_on_close)

    with pytest.raises(OSError, match="archive close failed"):
        src.close()
    src.close()

    assert raw.closed and src.closed and calls == [True]


# ==================================================
@pytest.mark.parametrize("kind", ["gz", "tar.gz"])
def test_input_path_closes_archive_on_error(tmp_path, monkeypatch, kind):
    import symclosestwannier.util.input_file as input_file_module

    archive = break_archive(tmp_path / "a.spn", kind, "truncated")

    opened = []
    open_compressed = input_file_module._open_compressed
    tar_open = tarfile.open

    def record_open_compressed(file_name):
        src = open_compressed(file_name)
        opened.append(src)
        return src

    def record_tar_open(*args, **kwargs):
        tf = tar_open(*args, **kwargs)
        opened.append(tf)
        return tf

    monkeypatch.setattr(input_file_module, "_open_compressed", record_open_compressed)
    monkeypatch.setattr(input_file_module.tarfile, "open", record_tar_open)

    with pytest.raises(SymCWInputError, match=re.escape(f"cannot read {archive}")):
        with input_path(str(tmp_path / "a.spn"), "spn"):
            pass

    # gz fails while copying, and tar.gz while listing the members.
    assert len(opened) == 1
    assert opened[0].closed


# ==================================================
def test_open_input_fileno(tmp_path):
    (tmp_path / "a.eig").write_text("1\n")
    compress(tmp_path / "a.eig", "gz")

    with open_input(str(tmp_path / "a.eig"), "eig") as fp:
        assert os.fstat(fp.fileno()).st_size == os.path.getsize(tmp_path / "a.eig.gz")


# ==================================================
@pytest.mark.parametrize(
    "error, expected",
    [(zlib.error("invalid distance too far back"), SymCWInputError), (OSError("disk error"), OSError)],
)
def test_input_path_tar_member_fails_while_reading(tmp_path, monkeypatch, error, expected):
    import symclosestwannier.util.input_file as input_file_module

    (tmp_path / "a.spn").write_bytes(b"data" * 1000)
    compress(tmp_path / "a.spn", "tar.gz")

    class FailingMember:
        """
        tar member which fails after the first read.
        """

        def __init__(self, member):
            self.member = member
            self.reads = 0

        def read(self, size=-1):
            self.reads += 1
            if self.reads > 1:
                raise error
            return self.member.read(16)

        def close(self):
            self.member.close()

    opened = {}
    tar_open = tarfile.open

    def record_tar_open(*args, **kwargs):
        tf = tar_open(*args, **kwargs)
        extractfile = tf.extractfile

        def failing_extractfile(member):
            opened["member"] = extractfile(member)
            return FailingMember(opened["member"])

        tf.extractfile = failing_extractfile
        opened["archive"] = tf
        return tf

    created = []
    mkstemp = input_file_module.tempfile.mkstemp

    def record_mkstemp(*args, **kwargs):
        fd, path = mkstemp(*args, **kwargs)
        created.append(path)
        return fd, path

    monkeypatch.setattr(input_file_module.tarfile, "open", record_tar_open)
    monkeypatch.setattr(input_file_module.tempfile, "mkstemp", record_mkstemp)

    with pytest.raises(expected) as e:
        with input_path(str(tmp_path / "a.spn"), "spn"):
            pass

    if expected is SymCWInputError:
        assert f"cannot read {tmp_path / 'a.spn'}.tar.gz, the file may be broken" in str(e.value)
        assert e.value.__cause__ is error
    else:
        # errors other than decompression errors are not changed.
        assert e.value is error
    assert opened["member"].closed and opened["archive"].closed
    assert len(created) == 1 and not os.path.exists(created[0])
