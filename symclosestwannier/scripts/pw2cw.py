"""
create Closest Wannier (CW) tight-binding (TB) model from input.
"""
import click

from symclosestwannier.cw.cw_creator import cw_creator
from symclosestwannier.util.header import cwin_header


# ================================================== pw2cw
@click.command()
@click.option("-i", "--input", is_flag=True, help="Show input format, and exit.")
@click.argument("seedname", nargs=-1)
def cmd(seedname, input):
    """
    create Closest Wannier (CW) tight-binding (TB) model from input.

        seedname : seedname for seedname.cwin file (w or w/o `.cwin`).
    """
    if input:
        click.echo(cwin_header)
        exit()

    if len(seedname) < 1:
        exit()
    else:
        seedname = seedname[0]

    seedname = seedname.replace(" ", "")
    seedname = seedname[:-5] if seedname[-5:] == ".cwin" else seedname

    cw_creator(seedname)


# ================================================== main
def main():
    cmd()


if __name__ == "__main__":
    main()
