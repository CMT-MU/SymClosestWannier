"""
exceptions for errors caused by input files or settings.

pw2cw/postcw show these errors as a one-line message without a traceback.
They also derive from the corresponding built-in exceptions, so that
existing `except ValueError` / `except FileNotFoundError` keep working.
"""


# ==================================================
class SymCWError(Exception):
    """
    base class of errors caused by input files or settings.
    """

    pass


# ==================================================
class SymCWInputError(SymCWError, ValueError):
    """
    invalid input settings or contents of input files.
    """

    pass


# ==================================================
class SymCWFileNotFoundError(SymCWError, FileNotFoundError):
    """
    required input file is not found.
    """

    # ==================================================
    def __init__(self, kind, file_name, hint=""):
        """
        Args:
            kind (str): kind of file, e.g. "eig".
            file_name (str): file name.
            hint (str, optional): additional message.
        """
        self.kind = kind
        self.file_name = file_name
        self.hint = hint

        msg = f"cannot find the {kind} file: {file_name}"
        if hint:
            msg += f" ({hint})"
        super().__init__(msg)

    # ==================================================
    def __reduce__(self):
        # keep the constructor arguments, e.g. for pickle in parallel processes.
        return (self.__class__, (self.kind, self.file_name, self.hint))
