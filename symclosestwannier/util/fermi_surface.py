"""
utility codes for lindhard function.
"""

import subprocess
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import gc

from symclosestwannier.util.utility import (
    fermi,
    fourier_transform_r_to_k,
    fourier_transform_r_to_k_vec,
    spin_zeeman_interaction,
)
from symclosestwannier.analyzer.get_response import utility_w0gauss

_num_proc = multiprocessing.cpu_count()


# ==================================================
def output_fermi_surface_eig(outdir, seedname, kpoints_2d, e, **kwargs):
    """
    output fermi surface.
    (only eigen values)

    Args:
        outdir (str): input and output files are found in this directory.
        seedname (str): seedname.
        k (str): k points.
        e (ndarray): eigen values.
        kwargs (dict, optional): key words for generate_band_gnuplot.
    """
    kmax_1 = np.max(kpoints_2d[:, :, 0])
    kmin_1 = np.min(kpoints_2d[:, :, 0])
    kmax_2 = np.max(kpoints_2d[:, :, 1])
    kmin_2 = np.min(kpoints_2d[:, :, 1])

    num_k_1, num_k_2 = kpoints_2d.shape[:2]

    emax = np.max(e)
    emin = np.min(e)
    num_wann = e.shape[1]

    fs = open(outdir + "/" + seedname + "_band_contour.txt", "w")
    fs.write("# k1 k2 E1 E2 ... E_num_wann [eV] \n")
    fs.write(f"# Emax = {str(emax)}\n")
    fs.write(f"# Emin = {str(emin)}\n")
    fs.write(f"# num_wann = {str(num_wann)}\n")
    fs.write(f"# (num_k_1, num_k_2) = {str((num_k_1, num_k_2))}\n")

    ef = kwargs.get("ef", None)

    if ef is None:
        ef = 0.0
        kwargs["ef"] = 0.0
        fs.write("# ef = ? (no shift) \n\n")
    else:
        fs.write(f"# shifted by fermi energy = {ef} [eV] \n\n")

    for i in range(num_k_1):
        for j in range(num_k_2):
            k = num_k_2 * i + j
            s = "{k1:0<20}   {k2:0<20}".format(k1=kpoints_2d[i, j, 0], k2=kpoints_2d[i, j, 1])

            for n in range(num_wann):
                s += "   {e:<20}".format(e=e[k, n] - ef)

            s += " \n"
            fs.write(s)

        s += " \n"
        fs.write(s)

    fs.close()

    # generate gnuplot file
    generate_fermi_surface_gnuplot_eig(outdir, seedname, kmax_1, kmin_1, kmax_2, kmin_2, emax, emin, num_wann, **kwargs)


# ==================================================
def generate_fermi_surface_gnuplot_eig(
    outdir, seedname, kmax_1, kmin_1, kmax_2, kmin_2, emax, emin, num_wann, **kwargs
):
    """
    generate gnuplot file to plot band dispersion.
    (only eigen values)

    Args:
        outdir (str): input and output files are found in this directory.
        seedname (str): seedname.
        kmax (float): maximum value in kpoints.
        emax (float): maximum value of eigen values.
        emin (float): minimum value of eigen values.
        num_wann (int): # of wannier functions.
        kwargs (dict, optional): key words for generate_band_gnuplot.
            - a (float): length of lattice vector.
            - ef (float): fermi energy.
            - k_dis_pos (dict): {disconnected linear position:label}.
            - lwidth (float): line width.
            - lc (str): line color.
            - ref_filename (str): file name of reference band data.
    """
    ef = kwargs.get("ef", 0.0)

    fs = open(f"{outdir}/plot_fermi_surface.gnu", "w")
    fs.write("set pm3d map \n")
    fs.write("unset key \n")
    fs.write("unset grid \n")
    fs.write("unset surface \n")
    fs.write("set size square  \n\n")
    fs.write(f"kmax_1 = {kmax_1} \n")
    fs.write(f"kmin_1 = {kmin_1} \n")
    fs.write(f"kmax_2 = {kmax_2} \n")
    fs.write(f"kmin_2 = {kmin_2} \n")
    fs.write("set xrange [kmin_1:kmax_1] \n")
    fs.write("set yrange [kmin_2:kmax_2] \n")
    fs.write(f"# ef = {ef} is set as zero.\n")
    fs.write("set tics font 'Times Roman, 20' \n\n")

    fs.write(f"emax = {emax - ef} \n")
    fs.write(f"emin = {emin - ef} \n")
    fs.write(f"emid = {((emax - ef) + (emin - ef))/2} \n")
    fs.write("set palette defined (emin 'royalblue', emid 'sea-green', emax 'salmon') \n\n")

    fs.write("set terminal postscript eps color enhanced \n\n")

    fs.write(f"do for[i=3:{num_wann + 2}]" + "{ \n")
    fs.write(f"set output sprintf('%s_%d.eps','{seedname}_band_contour', i) \n")
    fs.write(f"splot '{seedname}_band_contour.txt' u 1:2:i \n")
    fs.write("}")

    fs.write(" \n\n")

    fs.write("set terminal pdf \n\n")

    fs.write(f"do for[i=3:{num_wann + 2}]" + "{ \n")
    fs.write(f"set output sprintf('%s_%d.pdf','{seedname}_band_contour', i) \n")
    fs.write(f"splot '{seedname}_band_contour.txt' u 1:2:i \n ")
    fs.write("}")

    fs.write(" \n\n\n")

    # fermi surface
    fs.write("# output fermi surface  \n")
    fs.write("set contour  \n")
    fs.write("set cntrparam level discrete 0.0  \n")

    fs.write("set terminal postscript eps color enhanced \n\n")

    fs.write(f"set output '{seedname}_fermi_surface.eps' \n")
    fs.write(f"splot for [i=3:{num_wann + 2}] '{seedname}_band_contour.txt' u 1:2:i w l lw 1 lt 1 \n\n")

    fs.write("set terminal pdf \n\n")

    fs.write(f"set output '{seedname}_fermi_surface.pdf' \n")
    fs.write(f"splot for [i=3:{num_wann + 2}] '{seedname}_band_contour.txt' u 1:2:i w l lw 1 lt 1\n")

    fs.close()

    subprocess.run(f"cd {outdir} ; gnuplot plot_fermi_surface.gnu", shell=True)
