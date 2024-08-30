"""
utility codes for CW.
"""

import os
import subprocess
import numpy as np


# ==================================================
def _lorentzian(e, g=0.001):
    return (1.0 / np.pi) * g / (g**2 + e**2)


# ==================================================
def output_dos(outdir, filename, e, u, ef_shift=0.0, dos_num_fermi=50, dos_smr_en_width=0.001, **kwargs):
    """
    output density of states (DOS).

    Args:
        outdir (str): input and output files are found in this directory.
        filename (str): file name.
        e (ndarray): eigen values.
        u (ndarray): eigen vectors.
        ef_shift (float, optional): fermi energy shift, [0.0].
        dos_num_fermi (int, optional): number of fermi energies (int), [50].
        dos_smr_en_width (float): Energy width for the smearing function for the DOS (The units are [eV]) (flaot), [0.001].
        kwargs (dict, optional): key words for generate_dos_gnuplot.
    """
    emax = np.max(e)
    emin = np.min(e)
    offset = (emax - emin) * 0.1

    ef_max = emax + offset
    ef_min = emin - offset

    num_k, num_wann = e.shape
    dE = (ef_max - ef_min) / dos_num_fermi

    fermi_energy_list = [ef_min + i * dE for i in range(dos_num_fermi)]
    dos = (1.0 / num_k) * np.array([np.sum(_lorentzian(e - ef, dos_smr_en_width)) for ef in fermi_energy_list])

    u_abs2 = np.abs(u) ** 2
    pdos = (1.0 / num_k) * np.array(
        [
            [np.sum(u_abs2[:, m, :] * _lorentzian(e - ef, dos_smr_en_width)) for ef in fermi_energy_list]
            for m in range(num_wann)
        ]
    )

    filename = filename[:-4] if filename[-4:] == ".txt" else filename

    fs = open(outdir + "/" + filename + ".txt", "w")
    fs.write(f"# Fermi energy [eV] (shifted by ef = {ef_shift} [eV])  Total DOS [states/eV]  pDOS_1 ... pDOS_d \n")
    fs.write("# pDOS_j: partial DOS of jth orbital\n")
    fs.write(f"# fermi_energy_max = {str(emax)}\n")
    fs.write(f"# fermi_energy_min = {str(emin)}\n")
    fs.write(f"# num_wann = {str(num_wann)}\n")
    fs.write(f"# dos_num_fermi = {dos_num_fermi}\n")
    fs.write(f"# dos_smr_en_width = {dos_smr_en_width}\n")

    for i, ef in enumerate(fermi_energy_list):
        total_dos = dos[i]
        s = str(ef - ef_shift) + "  " + str(total_dos) + " "
        for m in range(num_wann):
            s += " " + str(pdos[m, i])

        s += "\n"
        fs.write(s)

    fs.close()

    dox_max = np.max(dos)

    # generate gnuplot file
    generate_dos_gnuplot(outdir, filename, emax, emin, ef_shift, dox_max, num_wann, **kwargs)


# ==================================================
def generate_dos_gnuplot(outdir, filename, emax, emin, ef_shift, dos_max, num_wann, **kwargs):
    """
    generate gnuplot file to plot dos.

    Args:
        outdir (str): input and output files are found in this directory.
        filename (str): file name.
        ef_max (float): maximum value of the fermi energy.
        ef_min (float): minimum value of the fermi energy.
        ef_shift (float): fermi energy shift.
        dos_max (float): maximum value of the DOS.
        num_wann (int): # of wannier functions.
        kwargs (dict, optional): key words for generate_dos_gnuplot.
            - a (float): length of lattice vector.
            - ef (float): fermi energy.
            - lwidth (float): line width.
            - lc (str): line color.
            - ref_filename (str): file name of reference band data.
    """
    offset = (emax - emin) * 0.1

    ef_max = emax + offset
    ef_min = emin - offset

    lwidth = kwargs.get("lwidth", 3)
    lc = kwargs.get("lc", "salmon")

    fs = open(f"{outdir}/plot_dos.gnu", "w")
    fs.write("unset key \n")
    fs.write("unset grid \n")
    fs.write(f"lwidth = {lwidth} \n")
    fs.write(f"set xrange [:{dos_max}] \n")
    fs.write(f"set yrange [{ef_min - ef_shift}:{ef_max - ef_shift}] \n")
    fs.write("set tics font 'Times Roman, 30' \n\n")
    fs.write("set size ratio 1.3 \n\n")
    fs.write(f"ef = {ef_shift} \n")

    fs.write("set terminal postscript eps color enhanced \n\n")

    fs.write(f"set output '{filename}.eps' \n\n")
    fs.write("plot ")

    fs.write(f"'{filename}.txt' u 2:1 w l lw lwidth dt (3,1) lc '{lc}', ")
    fs.write(f"{0.0} lw 0.5 dt (2,1) lc 'black'")

    fs.close()

    subprocess.run(f"cd {outdir} ; gnuplot plot_dos.gnu", shell=True)
