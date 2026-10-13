"""
CWManager manages current directory and formatter.
"""

import os
import sys
import codecs
import time
import ast
import pickle
import subprocess

from symclosestwannier.util.exceptions import SymCWFileNotFoundError


# ==================================================
def read_dict(filename):
    """
    read a dict from a text file written by write_dict (an optional docstring header and "var =" prefix).

    Args:
        filename (str): file name.

    Returns:
        dict: dictionary.
    """
    with open(filename, mode="r", encoding="utf-8") as f:
        body = ast.parse(f.read()).body

    # an optional docstring, then the dict as "var = {...}" or "{...}"
    if len(body) == 2 and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0].value, "value", None), str):
        body = body[1:]
    if len(body) != 1 or not (
        (isinstance(body[0], ast.Assign) and len(body[0].targets) == 1 and isinstance(body[0].targets[0], ast.Name))
        or isinstance(body[0], ast.Expr)
    ):
        raise ValueError(f"{filename} is not a dict file (an optional docstring and one dict).")

    dic = ast.literal_eval(body[0].value)
    if not isinstance(dic, dict):
        raise ValueError(f"{filename} is not a dict file (an optional docstring and one dict).")

    return dic


# ==================================================
def write_dict(filename, dic, header=None, var=None):
    """
    write a dict to a text file.

    Args:
        filename (str): file name.
        dic (dict): dictionary to write.
        header (str, optional): header comment (docstring) at the top of the file.
        var (str, optional): variable name of the dict.
    """
    with open(filename, mode="w", encoding="utf-8") as f:
        if header is not None:
            print('"""' + header + '"""', file=f)
        if var is None:
            print(dic, file=f)
        else:
            print(f"{var} =", dic, file=f)


# ==================================================
class CWManager:
    """
    manage directory, read, write etc.
    """

    # ==================================================
    def __init__(self, topdir=None, verbose=False, parallel=True, formatter=True):
        """
        manage directory, read, write etc.

        Args:
            topdir (str, optional): top directory for output, ends without slash.
            verbose (bool, optional): verbose parallel info. ?
            parallel (bool, optional): use parallel code ?
            formatter (bool, optional): format by using black ?
        """
        self._start = time.time()
        self._lap = self._start

        # initialize.
        if topdir is not None:
            os.makedirs(os.path.abspath(topdir), exist_ok=True)
            os.chdir(topdir)
            sys.path.append(os.getcwd())

        self._topdir = os.getcwd().replace(os.sep, "/")
        self._dirname = self._topdir

        self._verbose = verbose
        self._formatter = formatter
        self._parallel = parallel

    # ==================================================
    @property
    def dirname(self):
        return self._dirname

    # ==================================================
    @property
    def verbose(self):
        return self._verbose

    # ==================================================
    @property
    def formatter(self):
        return self._formatter

    # ==================================================
    @property
    def parallel(self):
        return self._parallel

    # ==================================================
    def elapsed(self, from_stamp=True):
        """
        elapsed time.

        Args:
            from_stamp (bool, optional): if True, return difference from lap.

        Returns:
            float: elapsed time.
        """
        if from_stamp:
            return time.time() - self._lap
        else:
            return time.time() - self._start

    # ==================================================
    def set_stamp(self):
        """
        set current time.
        """
        self._lap = time.time()

    # ==================================================
    def log(self, text, stamp="", end=None, file=None, mode="w"):
        """
        write log if verbose is True.

        Args:
            text (str): text to write.
            stamp (str or None, optional): attach elapsed time if "start" or "", otherwise no stamp is attached.
            end (str, optional): same as print end option.
            file (str, optional): same as print file option.
            mode (str, optional): mode, "w"/"a".
        """
        if stamp is not None:
            text += " ( " + str(round(self.elapsed(stamp != "start"), 3)) + " [sec] )."

        if self._verbose:
            print(text, end=end)

        if file is not None:
            file = codecs.open(file, mode, "utf-8")
            print(text, end=end, file=file)

    # ==================================================
    def filename(self, filename, full=True):
        """
        get (full) file name.

        Args:
            filename (str): file name.
            full (bool, optional): with full path ?

        Returns:
            str: full file name (with full path).
        """
        if full:
            if os.path.isabs(filename):  # e.g. in mp_outdir, which is an absolute path.
                return filename
            return self.dirname + "/" + filename
        else:
            return os.path.split(filename)[1]

    # ==================================================
    def read(self, file_dict):
        """
        read dict file or dict itself.

        Args:
            file_dict (str or dict): filename of dict. or dict.

        Returns:
            dict: read dict.
        """
        if isinstance(file_dict, str):
            full = self._topdir + "/" + file_dict
            if os.path.isfile(full):
                if "pkl" in full:
                    dic = pickle.load(open(full, "rb"))
                else:
                    dic = read_dict(full)
                self.log(f"  * read '{full}'.", None)
            else:
                raise SymCWFileNotFoundError("dict", full)
        else:
            dic = file_dict

        return dic

    # ==================================================
    def write(self, filename, dic, header=None, var=None):
        """
        write dict to file.

        Args:
            filename (str): file name.
            dic (dict): dict to write.
            header (str, optional): header of dict.
            var (str, optional): variable name for dict.
        """
        full = self.filename(filename)
        write_dict(full, dic, header, var)
        self.log(f"  * wrote '{filename}'.", None)

    # ==================================================
    def create_subdir(self, subdir):
        """
        create sub directory under topdir.
        """
        self._dirname = os.path.abspath(self._topdir + "/" + subdir)
        os.makedirs(self._dirname, exist_ok=True)

    # ==================================================
    def format_files(self):
        """
        format files in current output directory by using black, if formatter is True.
        """
        if self._formatter:
            cmd = "black --line-length=130 ."
            try:
                subprocess.run(cmd, capture_output=True, check=True, cwd=self._dirname, shell=True)
            except subprocess.CalledProcessError:
                raise RuntimeError("Formatting by black is failed.")
