"""
common part of pw2cw and postcw commands.
"""

import click

from symclosestwannier.util.header import cwin_header
from symclosestwannier.util.exceptions import SymCWError

from symclosestwannier.__init__ import __version__


# ==================================================
def run_command(ctx, func, seedname, input, version):
    """
    show input format or version, or run func(seedname).

    Args:
        ctx (click.Context): click context.
        func (callable): function to run with seedname.
        seedname (str or None): seedname for seedname.cwin file (w or w/o `.cwin`).
        input (bool): show input format, and exit.
        version (bool): show version, and exit.
    """
    if input:
        click.echo(cwin_header)
        ctx.exit()

    if version:
        click.echo(f"SymClosestWannier: {__version__}")
        ctx.exit()

    if seedname is None:
        raise click.UsageError("missing SEEDNAME.", ctx)

    seedname = seedname.replace(" ", "")
    seedname = seedname[:-5] if seedname.endswith(".cwin") else seedname

    # errors in input files or settings are shown without traceback.
    try:
        func(seedname)
    except SymCWError as e:
        raise click.ClickException(str(e)) from e
