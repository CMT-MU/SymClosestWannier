"""
utility codes for CW.
"""

import os
import subprocess
import numpy as np


# ==================================================
def output_linear_dispersion(outdir, filename, k, e, u, **kwargs):
    """
    output band dispersion along high-symmetry lines.

    Args:
        outdir (str): input and output files are found in this directory.
        filename (str): file name.
        k (str): k points along high-symmetry lines.
        e (ndarray): eigen values.
        u (ndarray): eigen vectors.
        kwargs (dict, optional): kwargs for generate_band_gnuplot.
    """
    kmax = np.max(k)
    emax = np.max(e)
    emin = np.min(e)
    num_wann = e.shape[1]

    filename = filename[:-4] if filename[-4:] == ".txt" else filename

    fs = open(outdir + "/" + filename + ".txt", "w")
    fs.write("# n = band, j = orbital, E_n: energy, W_jn: weight\n")
    fs.write("# k E_0 W_00 W_10 ... 1qW_M0 E_1 W_01 W_11 W_M1 ... E_n W_jn ... E_d W_0d W_1d ... W_dd\n")
    fs.write(f"# Emax = {str(emax)}\n")
    fs.write(f"# Emin = {str(emin)}\n")
    fs.write(f"# num_wann = {str(num_wann)}\n")

    ef = kwargs.get("ef", None)

    if ef is None:
        ef = 0.0
        kwargs["ef"] = 0.0
        fs.write("# ef = ? (no shift) \n\n")
    else:
        fs.write(f"# shifted by fermi energy = {ef} [eV] \n\n")

    num_k, Nm = e.shape
    e = e.T
    u = np.abs(u) ** 2

    for i in range(num_k):
        s = str(k[i])
        for n in range(Nm):
            en = e[n] - ef
            s += " " + str(en[i])
            for j in range(Nm):
                s += " " + str(u[i, j, n])
        s += "\n"
        fs.write(s)

    fs.close()

    # generate gnuplot file
    generate_band_gnuplot(outdir, filename, kmax, emax, emin, num_wann, **kwargs)


# ==================================================
def generate_band_gnuplot(outdir, filename, kmax, emax, emin, num_wann, **kwargs):
    """
    generate gnuplot file to plot band dispersion.

    Args:
        outdir (str): input and output files are found in this directory.
        filename (str): file name.
        kmax (float): maximum value in kpoints.
        emax (float): maximum value of eigen values.
        emin (float): minimum value of eigen values.
        num_wann (int): # of wannier functions.
        kwargs (dict, optional): kwargs.
            - a (float): length of lattice vector.
            - ef (float): fermi energy.
            - k_dis_pos (dict): {disconnected linear position:label}.
            - lwidth (float): line width.
            - lc (str): line color.
            - ref_filename (str): file name of reference band data.
    """
    offset = (emax - emin) * 0.1

    a = kwargs.get("a", None)
    ef = kwargs.get("ef", 0.0)
    k_dis_pos = kwargs.get("k_dis_pos", None)
    ref_filename = kwargs.get("ref_filename", None)
    lwidth = kwargs.get("lwidth", 3)
    lc = kwargs.get("lc", "salmon")

    fs = open(f"{outdir}/plot_band.gnu", "w")
    fs.write("unset key \n")
    fs.write("unset grid \n")
    fs.write(f"lwidth = {lwidth} \n")
    fs.write(f"set xrange [:{kmax}] \n")
    fs.write(f"set yrange [{emin-ef-offset}:{emax-ef+offset}] \n")
    fs.write("set tics font 'Times Roman, 30' \n\n")
    fs.write("set size ratio 0.7 \n\n")

    if k_dis_pos is not None:
        for pos, label in k_dis_pos.items():
            fs.write(f"set arrow from  {pos},  {emin-ef-offset} to  {pos}, {emax-ef+offset} nohead \n")

        k_dis_pos = {pos: "{/Symbol G}" if label == "G" else label for pos, label in k_dis_pos.items()}
        fs.write("set xtics (" + "".join([f"'{label}' {pos}," for pos, label in k_dis_pos.items()]) + ") \n\n")

    fs.write(f"ef = {ef} \n")

    if a is not None:
        fs.write(f"a = {a} \n")

    fs.write("set terminal postscript eps color enhanced \n\n")

    fs.write(f"set output '{filename}.eps' \n\n")
    fs.write("plot ")

    if ref_filename is not None:
        # fs.write(f"'{ref_filename}' u ($1/(2*pi)):2 w l lw lwidth lc 'dark-grey', ")
        fs.write(f"'{ref_filename}' u ($1/a):($2-ef) w l lw lwidth lc 'dark-grey', ")

    fs.write(
        f"for [j=2:{num_wann*(num_wann+1)}:{num_wann+1}] '{filename}.txt' u 1:j w l lw lwidth dt (3,1) lc '{lc}', "
    )

    fs.write(
        f"{0.0} lw 0.5 dt (2,1) lc 'black'"
    )

    fs.close()

    os.chdir(outdir)

    subprocess.run("gnuplot plot_band.gnu", shell=True)
