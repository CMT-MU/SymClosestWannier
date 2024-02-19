# ****************************************************************** #
#                                                                    #
# This file is distributed as part of the symclosestwannier code and #
#     under the terms of the GNU General Public License. See the     #
#     file LICENSE in the root directory of the symclosestwannier    #
#      distribution, or http://www.gnu.org/licenses/gpl-3.0.txt      #
#                                                                    #
#          The symclosestwannier code is hosted on GitHub:           #
#                                                                    #
#            https://github.com/CMT-MU/SymClosestWannier             #
#                                                                    #
#                            written by                              #
#                        Rikuto Oiwa, RIKEN                          #
#                                                                    #
# ------------------------------------------------------------------ #
#                                                                    #
#    pw2cw: create Closest Wannier tight-binding model from input    #
#                                                                    #
# ****************************************************************** #

import click

from symclosestwannier.cw.cw_creator import cw_creator
from symclosestwannier.util.header import cwin_header

from symclosestwannier.__init__ import __version__


# ================================================== pw2cw
@click.command()
@click.option("-i", "--input", is_flag=True, help="Show input format, and exit.")
@click.option("-v", "--version", is_flag=True, help="Show version, and exit.")
@click.argument("seedname", nargs=-1)
def cmd(seedname, input, version):
    """
    run pw2cw.

    create Closest Wannier tight-binding model from input.

        seedname : seedname for seedname.cwin file (w or w/o `.cwin`).
    """
    if input:
        click.echo(cwin_header)
        exit()

    if version:
        click.echo(f"SymClosestWannier: {__version__}")

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
