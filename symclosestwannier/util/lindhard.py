"""
utility codes for lindhard function.
"""

import time
import subprocess
import numpy as np
import multiprocessing
from joblib import Parallel, delayed, wrap_non_picklable_objects

from symclosestwannier.util.utility import (
    fermi,
    fourier_transform_r_to_k,
    fourier_transform_r_to_k_vec,
    spin_zeeman_interaction,
)
from symclosestwannier.analyzer.get_response import utility_w0gauss

_num_proc = multiprocessing.cpu_count()


# ==================================================
def get_lindhard(cwi, HH_R, qpoints, omega, ef, delta):
    """
    lind hard function.

    Args:
        cwi (CWInfo): CWInfo.
        HH_R (ndarray): Hamiltonian matrix.
        qpoints (ndarray): qpoints.
        omega (float): frequency.
        ef (float): fermi energy (The units are [eV]).
        delta (float): Energy width for the smearing function (The units are [eV]).

    Returns:
        ndarray: lindhard function.
    """
    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    num_wann = cwi["num_wann"]
    num_q = len(qpoints)

    # ==================================================
    # @wrap_non_picklable_objects
    def get_lindhard_k(kpt):
        """
        calculate lindhard function,
        separated into Hermitian (lindhard_H) and anti-Hermitian (lindhard_AH) parts.

        Args:
            kpt (ndarray): qpoint.

        Returns:
            ndarray: lindhard function.
        """
        if kpt.ndim == 1:
            kpt = np.array([kpt])

        num_k = len(kpt)
        kqpt = np.array([k + q for q in qpoints for k in kpt], dtype=float)

        HHk = fourier_transform_r_to_k(HH_R, kpt, cwi["irvec"], cwi["ndegen"], atoms_frac)
        HHkq = fourier_transform_r_to_k(HH_R, kqpt, cwi["irvec"], cwi["ndegen"], atoms_frac)

        if cwi["zeeman_interaction"]:
            B = cwi["magnetic_field"]
            theta = cwi["magnetic_field_theta"]
            phi = cwi["magnetic_field_phi"]
            g_factor = cwi["g_factor"]

            pauli_spin_k = fourier_transform_r_to_k_vec(operators["SS_R"], kpt, cwi["irvec"], cwi["ndegen"], atoms_frac)
            HHk += spin_zeeman_interaction(B, theta, phi, pauli_spin_k, g_factor, cwi["num_wann"])
            pauli_spin_kq = fourier_transform_r_to_k_vec(
                operators["SS_R"], kqpt, cwi["irvec"], cwi["ndegen"], atoms_frac
            )
            HHkq += spin_zeeman_interaction(B, theta, phi, pauli_spin_kq, g_factor, cwi["num_wann"])

        Ek, Uk = np.linalg.eigh(HHk)
        Ekq, Ukq = np.linalg.eigh(HHkq)

        HHk = HHkq = Uk = Ukq = None

        Ekq = Ekq.reshape((num_q, num_k, num_wann))
        ek_ekq = Ek[np.newaxis, :, np.newaxis, :] - Ekq[:, :, :, np.newaxis]
        occk = fermi(Ek - ef, T=0.0, unit="eV")
        occkq = fermi(Ekq - ef, T=0.0, unit="eV")

        Ek = Ekq = None

        numerator = occk[np.newaxis, :, np.newaxis, :] - occkq[:, :, :, np.newaxis]

        occk = occkq = None

        denominator = ek_ekq + omega - 1.0j * delta
        denominator /= (ek_ekq + omega) ** 2 + delta**2

        ek_ekq = None

        lindhard = -np.sum(numerator * denominator, axis=(1, 2, 3))

        return lindhard

    # ==================================================
    lindhard_kmesh = cwi["lindhard_kmesh"]
    N1, N2, N3 = lindhard_kmesh
    num_k = np.prod(lindhard_kmesh)

    kpoints = np.array(
        [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
    )
    kpoints_chunks = np.split(kpoints, [j for j in range(1000, len(kpoints), 1000)])
    num_chunks = len(kpoints_chunks)

    lindhard = np.zeros(num_q, dtype=complex)

    start_time = time.time()
    for i, kpoints in enumerate(kpoints_chunks):
        lindhard += get_lindhard_k(kpoints)

        progress_ratio = (i + 1) / num_chunks
        bar_length = 30
        num_stars = int(progress_ratio * bar_length)
        bar = "*" * num_stars + "-" * (bar_length - num_stars)
        print(f"\r[{bar}] ({i+1}/{num_chunks})", end="", flush=True)

    # convert second to hour, minute and seconds
    end_time = time.time()
    elapsed_time = int(end_time - start_time)
    elapsed_hour = elapsed_time // 3600
    elapsed_minute = (elapsed_time % 3600) // 60
    elapsed_second = elapsed_time % 3600 % 60

    # print as 00:00:00
    print(
        " ("
        + str(elapsed_hour).zfill(2)
        + ":"
        + str(elapsed_minute).zfill(2)
        + ":"
        + str(elapsed_second).zfill(2)
        + ")"
    )

    fac = 1.0 / num_k

    lindhard *= fac

    return lindhard


# ==================================================
def output_lindhard(outdir, filename, om, q, lindhard, **kwargs):
    """
    output lindhard function along high-symmetry lines.

    Args:
        outdir (str): input and output files are found in this directory.
        filename (str): file name.
        om (float): frequency.
        q (ndarray): q points along high-symmetry lines.
        lindhard (ndarray): lindhard function L(om, q).
        kwargs (dict, optional): key words for generate_band_gnuplot.
    """
    qmax = np.max(q)
    num_q = len(q)
    lmax = np.max([np.max(np.real(lindhard)), np.max(np.imag(lindhard))])
    lmin = np.min([np.min(np.real(lindhard)), np.min(np.imag(lindhard))])

    filename = filename[:-4] if filename[-4:] == ".txt" else filename

    fs = open(outdir + "/" + filename + ".txt", "w")
    fs.write("# q real imag \n")
    fs.write(f"# max = {str(lmax)}\n")
    fs.write(f"# min = {str(lmin)}\n")
    fs.write(f"# omega = {str(om)}\n")
    fs.write(f"# num_q = {str(num_q)}\n")

    ef = kwargs.get("ef", None)

    if ef is None:
        ef = 0.0
        kwargs["ef"] = 0.0
        fs.write("# ef = ? (no shift) \n\n")
    else:
        fs.write(f"# shifted by fermi energy = {ef} [eV] \n\n")

    for i in range(num_q):
        s = "{q:0<20}   {Lqr:<20}   {Lqi:<20} \n".format(q=q[i], Lqr=np.real(lindhard[i]), Lqi=np.imag(lindhard[i]))
        fs.write(s)

    fs.close()

    # generate gnuplot file
    generate_lindhard_gnuplot(outdir, filename, qmax, lmax, lmin, **kwargs)


# ==================================================
def generate_lindhard_gnuplot(outdir, filename, qmax, lmax, lmin, **kwargs):
    """
    generate gnuplot file to plot lindhard function.

    Args:
        outdir (str): input and output files are found in this directory.
        filename (str): file name.
        qmax (float): maximum value in qpoints.
        lmax (float): maximum value of lindhard function.
        lmin (float): minimum value of lindhard function.
        kwargs (dict, optional): key words for generate_lindhard_gnuplot.
            - q_dis_pos (dict): {disconnected linear position:label}.
            - lwidth (float): line width.
            - lc (str): line color.
    """
    offset = (lmax - lmin) * 0.1

    q_dis_pos = kwargs.get("q_dis_pos", None)
    lwidth = kwargs.get("lwidth", 3)
    lcr = kwargs.get("lcr", "salmon")
    lci = kwargs.get("lci", "royalblue")

    fs = open(f"{outdir}/plot_lindhard.gnu", "w")
    fs.write("unset key \n")
    fs.write("unset grid \n")
    fs.write(f"lwidth = {lwidth} \n")
    fs.write(f"set xrange [:{qmax}] \n")
    fs.write(f"set yrange [{lmin-offset}:{lmax+offset}] \n")
    fs.write("set tics font 'Times Roman, 30' \n\n")
    fs.write("set size ratio 0.7 \n\n")

    if q_dis_pos is not None:
        for pos, label in q_dis_pos.items():
            fs.write(f"set arrow from  {pos},  {lmin-offset} to  {pos}, {lmax+offset} nohead \n")

        q_dis_pos = {pos: "{/Symbol G}" if label == "G" else label for pos, label in q_dis_pos.items()}
        fs.write("set xtics (" + "".join([f"'{label}' {pos}," for pos, label in q_dis_pos.items()]) + ") \n\n")

    fs.write("set terminal postscript eps color enhanced \n\n")

    fs.write(f"set output '{filename}.eps' \n\n")
    fs.write("plot ")
    fs.write(f"'{filename}.txt' u 1:2 w l lw lwidth dt (3,1) lc '{lcr}' title 're', ")
    fs.write(f"'{filename}.txt' u 1:3 w l lw lwidth dt (3,1) lc '{lci}' title 'im', ")
    fs.write(f"{0.0} lw 0.5 dt (2,1) lc 'black'")

    fs.write(" \n\n")

    fs.write("set terminal pdf \n\n")

    fs.write(f"set output '{filename}.pdf' \n\n")
    fs.write("plot ")
    fs.write(f"'{filename}.txt' u 1:2 w l lw lwidth dt (3,1) lc '{lcr}' title 're', ")
    fs.write(f"'{filename}.txt' u 1:3 w l lw lwidth dt (3,1) lc '{lci}' title 'im', ")
    fs.write(f"{0.0} lw 0.5 dt (2,1) lc 'black'")

    fs.close()

    subprocess.run(f"cd {outdir} ; gnuplot plot_lindhard.gnu", shell=True)
