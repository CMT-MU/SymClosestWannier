"""
utility codes for CW.
"""

import os
import subprocess
import numpy as np

from symclosestwannier.util.utility import fermi, fourier_transform_r_to_k, convert_w90_orbital


# ==================================================
def _lorentzian(e, g=0.001):
    return (1.0 / np.pi) * g / (g**2 + e**2)


# ==================================================
def output_cohp(
    outdir,
    filename,
    Hr,
    Ek,
    Uk,
    kpoints,
    irvec,
    ndegen,
    atoms_frac,
    nw2n,
    nw2l,
    nw2m,
    nw2r,
    nw2s,
    A,
    cohp_bond_length_max,
    cohp_bond_length_min,
    cohp_head_atom,
    cohp_tail_atom,
    cohp_head_atom_idx,
    cohp_tail_atom_idx,
    cohp_num_fermi=50,
    cohp_smr_en_width=0.001,
    cohp_emax=None,
    cohp_emin=None,
    ef_shift=0.0,
    **kwargs,
):
    """
    output density of states (DOS).

    Args:
        outdir (str): input and output files are found in this directory.
        filename (str): file name.
        Hr (ndarray): Hamiltonian matrix in real space.
        kpoints,
        irvec (ndarray, optional): irreducible R vectors (crystal coordinate, [[n1,n2,n3]], nj: integer).
        ndegen (ndarray, optional): number of degeneracy at each R.
        atoms_frac,
        nw2n,
        nw2l,
        nw2m,
        nw2r,
        nw2s,
        A (list/ndarray): real lattice vectors, A = [a1,a2,a3] (list), [[[1,0,0], [0,1,0], [0,0,1]]].
        ef (float, optional): fermi energy, [0.0].
        ef_shift (float, optional): fermi energy shift for plot, [0.0].
        cohp_num_fermi (int, optional): number of fermi energies (int), [50].
        cohp_smr_en_width (float): Energy width for the smearing function for the DOS (The units are [eV]) (flaot), [0.001].
        cohp_emax (float, optional): maximun energy to be calculated.
        cohp_emin (float, optional): minimun energy to be calculated.
        cohp_bond_length_max (float, optional): maximum bond length (ang) to be calculated.
        cohp_bond_length_min (float, optional): minimum bond length (ang) to be calculated.
        kwargs (dict, optional): key words for generate_cohp_gnuplot.
    """
    Hr = np.array(Hr, dtype=complex)
    e = np.array(Ek, dtype=float)
    u = np.array(Uk, dtype=complex)

    emax = np.max(e) if cohp_emax is None else cohp_emax
    emin = np.min(e) if cohp_emin is None else cohp_emin
    offset = (emax - emin) * 0.1
    ef_max = emax + offset
    ef_min = emin - offset

    num_k = e.shape[0]
    num_wann = Hr.shape[1]
    dE = (ef_max - ef_min) / cohp_num_fermi
    fermi_energy_list = [ef_min + i * dE for i in range(cohp_num_fermi + 1)]

    num_R = len(irvec)
    atoms_list = list(atoms_frac.values())
    atoms_positions = np.array([atoms_list[i] for i in nw2n])

    iRmn_bv_list = []
    for iR in range(num_R):
        n1, n2, n3 = irvec[iR]
        R = np.array([n1, n2, n3]) @ A
        for m in range(num_wann):
            atom_m, idx_m = list(atoms_frac.keys())[nw2n[m]]
            if cohp_head_atom != atom_m or cohp_head_atom_idx != idx_m:
                continue
            rm = atoms_positions[m] @ A
            for n in range(num_wann):
                atom_n, idx_n = list(atoms_frac.keys())[nw2n[n]]
                if cohp_tail_atom != atom_n or cohp_tail_atom_idx != idx_n:
                    continue
                rn = atoms_positions[n] @ A
                bv = rm - (R + rn)
                bv_length = np.linalg.norm(bv)
                bv_length = "{:.16f}".format(bv_length)

                if cohp_bond_length_min <= float(bv_length) and float(bv_length) <= cohp_bond_length_max:
                    iRmn_bv_list.append((iR, m, n, bv_length))

    iRmn_bv_list = sorted(set(iRmn_bv_list), key=iRmn_bv_list.index)

    ndegen = np.asarray(ndegen, dtype=float)
    weight = 1.0 / ndegen

    kR = np.einsum("ka,Ra->kR", kpoints, irvec, optimize=True)
    phase_R = np.exp(-2 * np.pi * 1j * kR)

    def _convert_w90_orbital(m):
        return convert_w90_orbital(nw2l[m], nw2m[m], nw2r[m], nw2s[m])

    pcohp = []
    print("")
    for ief in range(cohp_num_fermi + 1):
        print(f"{ief+1}/{cohp_num_fermi + 1}")
        ef = fermi_energy_list[ief]
        delta_func = _lorentzian(e - ef, cohp_smr_en_width)
        delta_func = np.array([np.diag(delta_func_k) for delta_func_k in delta_func], dtype=float)
        delta_func = u.transpose(0, 2, 1).conjugate() @ delta_func @ u
        nr_delta_func = np.einsum("kR,kmn->Rmn", phase_R, delta_func, optimize=True) / num_k

        pcohp.append(
            {
                (_convert_w90_orbital(m), _convert_w90_orbital(n), bv_length): np.real(
                    weight[iR] ** 2 * Hr[iR, m, n] * np.conjugate(nr_delta_func[iR, m, n])
                )
                for iR, m, n, bv_length in iRmn_bv_list
            }
        )

    pcohp_integrated = []
    for ief in range(cohp_num_fermi + 1):
        d_integated = {k: 0.0 for k in pcohp[0].keys()}
        for d in pcohp[: ief + 1]:
            for k, v in d.items():
                d_integated[k] += v * dE
        pcohp_integrated.append(d_integated)

    filename = filename[:-4] if filename[-4:] == ".txt" else filename

    fs = open(outdir + "/" + filename + ".txt", "w")
    fs.write(
        f"# Fermi energy [eV] (shifted by ef = {ef_shift} [eV]) cohp icohp pcohp_1 ipcohp_1 pcohp_2 ipcohp_2 ... pcohp_N ipcohp_N \n"
    )
    fs.write("# cohp: COHP of given atom pair\n")
    fs.write("# icohp: integrated COHP of given atom pair\n")
    fs.write("# pcohp_j: pCOHP of jth atom&orbital pair\n")
    fs.write("# ipcohp_j: integrated pCOHP of jth atom&orbital pair\n")
    fs.write(f"# fermi_energy_max = {str(ef_max)}\n")
    fs.write(f"# fermi_energy_min = {str(ef_min)}\n")
    fs.write(f"# num_wann = {str(num_wann)}\n")
    fs.write(f"# cohp_num_fermi = {cohp_num_fermi + 1}\n")
    fs.write(f"# cohp_smr_en_width = {cohp_smr_en_width}\n")
    fs.write(f"# num_k = {num_k}\n")
    fs.write(f"# cohp_bond_length_max = {cohp_bond_length_max}\n")
    fs.write(f"# cohp_bond_length_min = {cohp_bond_length_min}\n")
    fs.write(f"# cohp_head_atom = {cohp_head_atom}\n")
    fs.write(f"# cohp_tail_atom = {cohp_tail_atom}\n")
    fs.write(f"# cohp_head_atom_idx = {cohp_head_atom_idx}\n")
    fs.write(f"# cohp_tail_atom_idx = {cohp_tail_atom_idx}\n")

    orbmn_bv_list = list(pcohp[0].keys())
    for i, (head_orb, tail_orb, bv_length) in enumerate(orbmn_bv_list):
        fs.write(
            f"# No.{i+1}:{cohp_head_atom}_{cohp_head_atom_idx}{head_orb}<-{cohp_tail_atom}_{cohp_tail_atom_idx}{tail_orb}({bv_length}) \n"
        )

    for ief, ef in enumerate(fermi_energy_list):
        d = pcohp[ief]
        d_integated = pcohp_integrated[ief]
        cohp = np.sum([v for v in d.values()])
        icohp = np.sum([v_integrated for v_integrated in d_integated.values()])

        s = str(ef - ef_shift) + "  " + str(cohp) + "  " + str(icohp)

        for v, v_integrated in zip(d.values(), d_integated.values()):
            s += "  " + str(v) + "  " + str(v_integrated)

        s += "\n"
        fs.write(s)

    fs.close()

    cohp_max = np.max([np.sum([v for v in d.values()]) for d in pcohp])
    cohp_min = np.min([np.sum([v for v in d.values()]) for d in pcohp])
    icohp_max = np.max([np.sum([v for v in d.values()]) for d in pcohp_integrated])
    icohp_min = np.min([np.sum([v for v in d.values()]) for d in pcohp_integrated])

    # generate gnuplot file
    generate_cohp_gnuplot(
        outdir, filename, emax, emin, ef_shift, cohp_max, cohp_min, icohp_max, icohp_min, num_wann, **kwargs
    )


# ==================================================
def generate_cohp_gnuplot(
    outdir, filename, emax, emin, ef_shift, cohp_max, cohp_min, icohp_max, icohp_min, num_wann, **kwargs
):
    """
    generate gnuplot file to plot dos.

    Args:
        outdir (str): input and output files are found in this directory.
        filename (str): file name.
        ef_max (float): maximum value of the fermi energy.
        ef_min (float): minimum value of the fermi energy.
        ef_shift (float): fermi energy shift.
        cohp_max (float): maximum value of the DOS.
        num_wann (int): # of wannier functions.
        kwargs (dict, optional): key words for generate_cohp_gnuplot.
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

    fs = open(f"{outdir}/plot_cohp.gnu", "w")
    fs.write("unset key \n")
    fs.write("unset grid \n")
    fs.write(f"lwidth = {lwidth} \n")
    fs.write(f"set xrange [{-cohp_max}:{-cohp_min}] \n")
    fs.write(f"set yrange [{ef_min - ef_shift}:{ef_max - ef_shift}] \n")
    fs.write("set tics font 'Times New Roman, 24' \n\n")
    fs.write("set size ratio 1.3 \n\n")
    fs.write(f"ef = {ef_shift} \n")

    fs.write("set terminal postscript eps color enhanced \n\n")

    fs.write(f"set output '{filename}.eps' \n\n")
    fs.write("plot ")
    fs.write(f"'{filename}.txt' u (-$2):1 w l lw lwidth dt (3,1) lc '{lc}', ")
    fs.write(f"{0.0} lw 0.5 dt (2,1) lc 'black'")

    fs.write(" \n\n")

    fs.write("set terminal pdf \n")
    fs.write(f"set output '{filename}.pdf' \n")
    fs.write("replot")

    fs.write(" \n\n")

    fs.write(f"set xrange [{-icohp_max}:{-icohp_min}] \n")

    fs.write("set terminal postscript eps color enhanced \n\n")

    fs.write(f"set output '{filename.replace('cohp', 'icohp')}.eps' \n\n")
    fs.write("plot ")

    fs.write(f"'{filename}.txt' u (-$3):1 w l lw lwidth dt (3,1) lc '{lc}', ")
    fs.write(f"{0.0} lw 0.5 dt (2,1) lc 'black'")

    fs.write(" \n\n")

    fs.write("set terminal pdf \n\n")
    fs.write(f"set output '{filename.replace('cohp', 'icohp')}.pdf' \n\n")
    fs.write("replot")

    fs.close()

    subprocess.run(f"cd {outdir} ; gnuplot plot_cohp.gnu", shell=True)
