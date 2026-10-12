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
#              postcw: analyze Closest Wannier TB model              #
#                                                                    #
# ****************************************************************** #

import click

from symclosestwannier.analyzer.analyzer import analyzer
from symclosestwannier.scripts.common import run_command


# ================================================== postcw
@click.command()
@click.option("-i", "--input", is_flag=True, help="Show input format, and exit.")
@click.option("-v", "--version", is_flag=True, help="Show version, and exit.")
@click.argument("seedname", required=False)
@click.pass_context
def cmd(ctx, seedname, input, version):
    """
    run postcw.

        seedname : seedname for seedname.cwin file (w or w/o `.cwin`).
    """
    run_command(ctx, analyzer, seedname, input, version)


# ================================================== main
def main():
    cmd()


if __name__ == "__main__":
    main()
