"""
open input files given as file_name, file_name.gz or file_name.tar.gz.
"""

import os
import io
import gzip
import zlib
import shutil
import tarfile
import tempfile
import contextlib

from symclosestwannier.util.exceptions import SymCWFileNotFoundError, SymCWInputError

# errors raised while reading a broken compressed file.
_DECOMPRESS_ERRORS = (EOFError, zlib.error, gzip.BadGzipFile, tarfile.TarError)


# ==================================================
@contextlib.contextmanager
def _check_archive(archive):
    """
    raise SymCWInputError with the file name for errors of a broken compressed file.

    Args:
        archive (str): file name of the compressed file.
    """
    try:
        yield
    except _DECOMPRESS_ERRORS as e:
        raise SymCWInputError(f"cannot read {archive}, the file may be broken ({type(e).__name__}: {e}).") from e


# ==================================================
class _CheckedReader(io.BufferedIOBase):
    """
    binary stream of a compressed file, which reports a broken file as SymCWInputError.
    decompression errors can occur at any read, not only when the file is opened.
    """

    def __init__(self, raw, archive, on_close=None):
        """
        Args:
            raw (file object): binary stream of decompressed data.
            archive (str): file name of the compressed file.
            on_close (callable, optional): called after raw is closed, e.g. to close the tar archive.
        """
        self._raw = raw
        self._archive = archive
        self._on_close = on_close

    @property
    def name(self):
        return self._archive

    def readable(self):
        return True

    def seekable(self):
        return self._raw.seekable()

    def seek(self, offset, whence=io.SEEK_SET):
        # seeking backwards decompresses the data again.
        with _check_archive(self._archive):
            return self._raw.seek(offset, whence)

    def tell(self):
        return self._raw.tell()

    def fileno(self):
        return self._raw.fileno()

    def read(self, size=-1):
        with _check_archive(self._archive):
            return self._raw.read(size)

    def read1(self, size=-1):
        with _check_archive(self._archive):
            return self._raw.read1(size)

    def readinto(self, b):
        with _check_archive(self._archive):
            return self._raw.readinto(b)

    def close(self):
        if self.closed:
            return
        try:
            self._raw.close()
        finally:
            try:
                if self._on_close is not None:
                    self._on_close()
            finally:
                super().close()


# ==================================================
def _tar_member(tf, file_name, archive):
    """
    member of tar archive for file_name: the file with the same base name, or the only file in the archive.

    Args:
        tf (tarfile.TarFile): tar archive.
        file_name (str): file name without .tar.gz.
        archive (str): file name of the archive.

    Returns:
        tarfile.TarInfo: member.
    """
    with _check_archive(archive):
        files = [m for m in tf.getmembers() if m.isfile()]
    same_name = [m for m in files if os.path.basename(m.name) == os.path.basename(file_name)]

    if len(same_name) == 1:
        return same_name[0]
    if len(files) == 1:
        return files[0]

    raise SymCWInputError(f"cannot find {os.path.basename(file_name)} in {archive}.")


# ==================================================
def _open_compressed(file_name):
    """
    open file_name.gz or file_name.tar.gz as a binary stream (in this order of priority).

    Args:
        file_name (str): file name without .gz or .tar.gz.

    Returns:
        _CheckedReader or None: binary stream, or None if there is no compressed file.
    """
    if os.path.exists(file_name + ".gz"):
        archive = file_name + ".gz"
        return _CheckedReader(gzip.open(archive, "rb"), archive)

    if os.path.exists(file_name + ".tar.gz"):
        archive = file_name + ".tar.gz"
        with _check_archive(archive):
            tf = tarfile.open(archive, "r:gz")
        try:
            member = _tar_member(tf, file_name, archive)
            with _check_archive(archive):
                raw = tf.extractfile(member)
            return _CheckedReader(raw, archive, on_close=tf.close)
        except BaseException:
            tf.close()
            raise

    return None


# ==================================================
def open_input(file_name, kind):
    """
    open file_name, file_name.gz or file_name.tar.gz as a text file (in this order of priority).

    Args:
        file_name (str): file name.
        kind (str): kind of file used in the error message, e.g. "eig".

    Returns:
        file object: text file object, to be closed by the caller.
    """
    if os.path.exists(file_name):
        return open(file_name, "r")

    src = _open_compressed(file_name)
    if src is None:
        raise SymCWFileNotFoundError(kind, file_name)

    return io.TextIOWrapper(src)


# ==================================================
@contextlib.contextmanager
def input_path(file_name, kind):
    """
    path of file_name, file_name.gz or file_name.tar.gz as an uncompressed file.
    compressed files are extracted to a temporary file, which is removed afterwards.
    used for libraries which need a file name, e.g. fortio for unformatted files.

    Args:
        file_name (str): file name.
        kind (str): kind of file used in the error message, e.g. "spn".

    Yields:
        str: path of the uncompressed file.
    """
    if os.path.exists(file_name):
        yield file_name
        return

    src = _open_compressed(file_name)
    if src is None:
        raise SymCWFileNotFoundError(kind, file_name)

    # resources are closed (and the temporary file is removed) in reverse order, also on errors.
    with contextlib.ExitStack() as stack:
        stack.enter_context(src)

        fd, tmp = tempfile.mkstemp(prefix=os.path.basename(file_name) + ".")
        stack.callback(os.remove, tmp)
        try:
            dst = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with dst:
            shutil.copyfileobj(src, dst)

        yield tmp
