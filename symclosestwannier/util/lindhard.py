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
def get_lindhard(cwi, HH_R, qpoints, omega, ef, T, delta):
    """
    lind hard function.

    Args:
        cwi (CWInfo): CWInfo.
        HH_R (ndarray): Hamiltonian matrix.
        qpoints (ndarray): qpoints.
        omega (float): frequency.
        ef (float): fermi energy (The units are [eV]).
        temperature (float): temperature.
        delta (float): Energy width for the smearing function (The units are [eV]).

    Returns:
        ndarray: lindhard function.
    """

    HH_R = np.array(HH_R, dtype=np.complex64)

    def process_kpoint_chunk(kpoints, q, ef, HH_R, irvec, ndegen, atoms_frac, delta):
        lindhard_re_q = 0.0
        lindhard_im_q = 0.0

        HHk = fourier_transform_r_to_k(HH_R, kpoints, irvec, ndegen, atoms_frac)
        Ek, _ = np.linalg.eigh(HHk)

        kqpoints = kpoints + q
        HHkq = fourier_transform_r_to_k(HH_R, kqpoints, irvec, ndegen, atoms_frac)
        Ekq, _ = np.linalg.eigh(HHkq)

        delta_E = Ek[:, :, None] - Ekq[:, None, :]
        safe_delta_E = np.where(delta_E == 0, np.nan, delta_E)

        Ek_prod_Ekq = (Ek - ef)[:, :, None] * (Ekq - ef)[:, None, :]
        mask = Ek_prod_Ekq < 0

        occk = fermi(Ek - ef, T, unit="eV")
        occkq = fermi(Ekq - ef, T, unit="eV")
        delta_occ = occk[:, :, None] - occkq[:, None, :]

        contrib = -delta_occ / safe_delta_E
        contrib[~mask] = 0
        lindhard_re_q += np.nansum(contrib)

        deltak = utility_w0gauss((Ek - ef) / delta, n=0) / delta
        deltakq = utility_w0gauss((Ekq - ef) / delta, n=0) / delta

        lindhard_im_q += np.sum(deltak.T @ deltakq)

        gc.collect()

        return lindhard_re_q, lindhard_im_q

    # ==================================================

    if cwi["tb_gauge"]:
        atoms_list = list(cwi["atoms_frac"].values())
        atoms_frac = np.array([atoms_list[i] for i in cwi["nw2n"]])
    else:
        atoms_frac = None

    lindhard_kmesh = cwi["lindhard_kmesh"]
    N1, N2, N3 = lindhard_kmesh
    num_k = np.prod(lindhard_kmesh)

    kpoints = np.array(
        [[i / float(N1), j / float(N2), k / float(N3)] for i in range(N1) for j in range(N2) for k in range(N3)]
    )
    num_chunks = int(np.ceil(len(kpoints) / 10000))
    kpoints_chunks = np.array_split(kpoints, num_chunks)

    num_q = len(qpoints)
    lindhard_re = np.zeros(num_q, dtype=np.float32)
    lindhard_im_om0 = np.zeros(num_q, dtype=np.float32)

    for q in tqdm(range(num_q)):
        qvec = qpoints[q]
        results_q = Parallel(n_jobs=_num_proc)(
            delayed(process_kpoint_chunk)(kpoints_chunk, qvec, ef, HH_R, cwi["irvec"], cwi["ndegen"], atoms_frac, delta)
            for kpoints_chunk in kpoints_chunks
        )
        re_q, im_q = zip(*results_q)
        lindhard_re[q] = sum(re_q)
        lindhard_im_om0[q] = sum(im_q)

    fac = 1.0 / num_k

    lindhard_re *= fac
    lindhard_im_om0 *= fac

    return lindhard_re, lindhard_im_om0


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


# ==================================================
def output_lindhard_surface(outdir, filename, om, qpoints_2d, lindhard_re, lindhard_im_om0, **kwargs):
    """
    output lindhard function (2d plane).

    Args:
        outdir (str): input and output files are found in this directory.
        filename (str): file name.
        om (float): frequency.
        qpoints_2d (ndarray): q points.
        lindhard_re (ndarray): real part of lindhard function L(om, q).
        lindhard_im_om0 (ndarray): imaginary part of lindhard function lim_{om->0} L(om, q)/om
        kwargs (dict, optional): key words for generate_band_gnuplot.
    """
    qmax_1 = np.max(qpoints_2d[:, :, 0])
    qmin_1 = np.min(qpoints_2d[:, :, 0])
    qmax_2 = np.max(qpoints_2d[:, :, 1])
    qmin_2 = np.min(qpoints_2d[:, :, 1])

    num_q_1, num_q_2 = qpoints_2d.shape[:2]

    lmax_re = np.max([np.max(np.real(lindhard_re)), np.max(np.imag(lindhard_re))])
    lmin_re = np.min([np.min(np.real(lindhard_re)), np.min(np.imag(lindhard_re))])
    lmax_im = np.max([np.max(np.real(lindhard_im_om0)), np.max(np.imag(lindhard_im_om0))])
    lmin_im = np.min([np.min(np.real(lindhard_im_om0)), np.min(np.imag(lindhard_im_om0))])

    filename = filename[:-4] if filename[-4:] == ".txt" else filename

    fs = open(outdir + "/" + filename + ".txt", "w")
    fs.write("# q1 q2 real imag \n")
    fs.write(f"# max_re = {str(lmax_re)}\n")
    fs.write(f"# min_re = {str(lmin_re)}\n")
    fs.write(f"# max_im_om0 = {str(lmax_im)}\n")
    fs.write(f"# min_im_om0 = {str(lmin_im)}\n")
    fs.write(f"# omega = {str(om)}\n")
    fs.write(f"# (num_q_1, num_q_2) = {str((num_q_1, num_q_2))}\n")

    ef = kwargs.get("ef", None)

    if ef is None:
        ef = 0.0
        kwargs["ef"] = 0.0
        fs.write("# ef = ? (no shift) \n\n")
    else:
        fs.write(f"# shifted by fermi energy = {ef} [eV] \n\n")

    for i in range(num_q_1):
        for j in range(num_q_2):
            n = num_q_2 * i + j
            s = "{q1:0<20}    {q2:0<20}   {Lqr:<20}   {Lqi:<20} \n".format(
                q1=qpoints_2d[i, j, 0],
                q2=qpoints_2d[i, j, 1],
                Lqr=np.real(lindhard_re[n]),
                Lqi=np.real(lindhard_im_om0[n]),
            )
            fs.write(s)
        fs.write("\n")

    fs.close()

    # generate gnuplot file
    generate_lindhard_surface_gnuplot(
        outdir, filename, qmax_1, qmin_1, qmax_2, qmin_2, lmax_re, lmin_re, lmax_im, lmin_im, **kwargs
    )


# ==================================================
def generate_lindhard_surface_gnuplot(
    outdir, filename, qmax_1, qmin_1, qmax_2, qmin_2, lmax_re, lmin_re, lmax_im, lmin_im, **kwargs
):
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
    fs = open(f"{outdir}/plot_lindhard_surface.gnu", "w")
    fs.write("set pm3d  \n")
    fs.write("set pm3d map \n")
    fs.write("set ticslevel 0 \n")
    fs.write("unset key \n")
    fs.write("unset grid \n")
    fs.write("unset surface \n")
    fs.write(f"qmax_1 = {qmax_1} \n")
    fs.write(f"qmin_1 = {qmin_1} \n")
    fs.write(f"qmax_2 = {qmax_2} \n")
    fs.write(f"qmin_2 = {qmin_2} \n")
    fs.write("set xrange [qmin_1:qmax_1] \n")
    fs.write("set yrange [qmin_2:qmax_2] \n")
    fs.write("set tics font 'Times Roman, 20' \n\n")

    fs.write(f"lmax_re = {lmax_re} \n")
    fs.write(f"lmin_re = {lmin_re} \n")
    fs.write(f"lmid_re = {(lmax_re + lmin_re)/2} \n")
    fs.write("set palette defined (lmin_re 'royalblue',lmid_re 'sea-green', lmax_re 'salmon') \n\n")

    fs.write("set terminal postscript eps color enhanced \n\n")

    fs.write(f"set output '{filename}_re.eps' \n\n")
    fs.write("splot ")
    fs.write(f"'{filename}.txt' u 1:2:3, ")

    fs.write(" \n\n")

    fs.write("set terminal pdf \n\n")

    fs.write(f"set output '{filename}_re.pdf' \n\n")
    fs.write("splot ")
    fs.write(f"'{filename}.txt' u 1:2:3, ")

    fs.write(f"lmax_im = {lmax_im} \n")
    fs.write(f"lmin_im = {lmin_im} \n")
    fs.write(f"lmid_im = {(lmax_im + lmin_im)/2} \n")
    fs.write("set palette defined (lmin_im 'royalblue',lmid_im 'sea-green', lmax_im 'salmon') \n\n")

    fs.write("set terminal postscript eps color enhanced \n\n")

    fs.write(f"set output '{filename}_im_om0.eps' \n\n")
    fs.write("splot ")
    fs.write(f"'{filename}.txt' u 1:2:4, ")

    fs.write(" \n\n")

    fs.write("set terminal pdf \n\n")

    fs.write(f"set output '{filename}_im_om0.pdf' \n\n")
    fs.write("splot ")
    fs.write(f"'{filename}.txt' u 1:2:4, ")

    fs.close()

    subprocess.run(f"cd {outdir} ; gnuplot plot_lindhard_surface.gnu", shell=True)
