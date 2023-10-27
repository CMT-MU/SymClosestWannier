"""
create Symmetry-adapted Closest Wannier (SymCW) tight-binding model from input.
"""
import click

from symclosestwannier.pw2scw.create_scw import create_scw
from symclosestwannier.util.header import input_header_str


# ================================================== pw2scw
@click.command()
@click.option("-i", "--input", is_flag=True, help="Show input format, and exit.")
@click.argument("seedname", nargs=-1)
def cmd(seedname, input):
    """
    create Symmetry-adapted Closest Wannier (SymCW) tight-binding model from input.

        seedname : seedname for seedname.cwin file (w or w/o `.cwin`).
    """
    if input:
        click.echo(input_header_str)
        exit()

    if len(seedname) < 1:
        exit()
    else:
        seedname = seedname[0]

    seedname = seedname.replace(" ", "")
    seedname = seedname[:-5] if seedname[-5:] == ".cwin" else seedname

    create_scw(seedname)


# ================================================== main
def main():
    cmd()


if __name__ == "__main__":
    main()
