# Specification: the response module of postcw, to be written from the papers

Status: draft for review, revised after the design review (2026-10-27).

## 0. Purpose and rules

`symclosestwannier/analyzer/get_response.py` computes the responses of `postcw` (anomalous Hall conductivity, orbital
magnetization, optical conductivity, shift current, spin Hall conductivity, the K tensor of the gyrotropic effects). Most
of it was ported from wannier90's postw90 (`berry.F90`, `wan_ham.F90`, `gyrotropic.F90`, `utility.F90`), which was
distributed under the GPL. To distribute SymClosestWannier under the Mozilla Public License 2.0, the module is replaced by
a new implementation written from the equations of the papers.

Rules for whoever writes the new implementation:

1. **Do not read** `symclosestwannier/analyzer/get_response.py` (any version, including git history), the source code of
   wannier90 (Fortran), or the source code of other programs that implement these quantities (e.g. wannier-berri).
   The documentation of wannier90's input parameters (user guide) may be read, for the meaning of the input keys only.
   The implementer works in a copy of the repository **without `.git` and without `get_response.py`** and without the
   tests that call its internal functions (section 7.1, item 3), so that nothing can be opened by accident. Strip:
   `.git/`, `.ai-review/`, `symclosestwannier/analyzer/get_response.py`, `tests/test_tb_gauge.py`,
   `tests/test_berry_morb_sc.py`, `tests/test_gyrotropic.py`, `tests/test_smearing.py`, `tests/test_kubo_frequencies.py`,
   `tests/test_shc_bandshift.py`, `tests/test_fermi_energy.py`. Run the reference test with
   `pytest tests/test_response_reference.py`; never regenerate the reference values (`python tests/response_reference.py
   --write`).
2. Work from this specification and from the papers in section 8. **Acceptance is reproducing the reference values**
   (section 7.1, backward compatibility). If a formula derived from a paper disagrees with the reference values, do not
   change either silently: report the disagreement with the derivation, and the maintainers decide.
3. The modules of SymClosestWannier listed in section 2.4 may be read and used.
4. Record in the pull request which equation of which paper each function implements (a docstring line per function,
   e.g. `Eq. (32) of WYSV06`).

**Provenance of this specification.** Its author had read the old module and wannier90 (for the audit of what is
derived from wannier90). The specification therefore restricts itself to
- (P) equations of the papers, cited by number (the numbers come from citations in the old sources and are not
  verified: confirm each one in the paper and cite what the paper says);
- (B) black-box behaviour of the public entry points (inputs, outputs, shapes, units, errors, normalization constants),
  which is the interface to be kept and is marked [B];
- (I) interfaces of SymClosestWannier's own modules.
It contains no code of the old module or of wannier90. The pull request of the implementation keeps this
classification for every requirement it relies on.

## 1. Scope

The new module replaces `get_response.py` as a whole, with the same module name and public functions (section 2.1).
`spin_moment_main` (spin magnetic moment, with the Zeeman term) and `_single_fermi_energy` are SymClosestWannier's own
code; they have been moved unchanged to `analyzer/spin_moment.py`, which the new module re-exports. That file keeps its
GPL header until its authors confirm the relicensing (open item for the maintainers, not for the implementer). Out of
scope: the construction of the real-space operators (`util/get_oper_R.py`; its relicensing is a separate decision of the
maintainers), the Fourier transforms and other helpers of `util/utility.py`, the input parsing (`cw/win.py`), and the
output (`analyzer/response.py`).

## 2. Interface

### 2.1 Public functions [B]

`analyzer/response.py` imports `berry_main`, `gyrotropic_main`, `spin_moment_main`, `kubo_frequencies` and
`boltzwann_main` from `symclosestwannier.analyzer.get_response`; `_single_fermi_energy` is also imported by tests. The
new module provides all of them (`spin_moment_main` and `_single_fermi_energy` by re-export, `boltzwann_main(*args, **kwargs)`
doing nothing and returning `None`; BoltzWann is not implemented).

`util/lindhard.py`, `util/fermi_surface.py` and `util/spread.py` import `utility_w0gauss(x, n)` from the module at import
time, so `pw2cw` and `postcw` fail without it. The new module provides it: the kernel w(x) of section 6 for a scalar or
an array `x`, returned as an ndarray of the shape of `x`; n = N with 0 ≤ N ≤ 10 is Methfessel–Paxton of order N (n = 0 is
gauss), n = −1 cold, n = −99 Fermi–Dirac (0 for |x| > 36) [B]; any other n raises `NotImplementedError` [B].

`berry_main(cwi, operators)`, `gyrotropic_main(cwi, operators)` and `spin_moment_main(cwi, operators)` print a short
progress log to stdout and return a dict. `cwi` is a dict-like object with the keys of section 2.2 and an attribute
`cwi.win`; `operators` is a dict of real-space operators (section 2.3).

**Returned keys** (the set of keys is part of the interface; the reference values check it):

- `berry_main`: always the keys `ahc_list`, `morb`, `kubo`, `kubo_H`, `kubo_AH`, `kubo_H_spn`, `kubo_AH_spn`, `sc`,
  `shc`, with value `None` unless set by the task `cwi["berry_task"]`:
  `"ahc"` sets `ahc_list`; `"morb"` sets `morb` and adds `morb_LC`, `morb_IC`; `"kubo"` sets `kubo_H`, `kubo_AH`, and
  `kubo_H_spn = kubo_AH_spn = 0.0` (Python float); `"sc"` sets `sc` and adds `sc_freq`; `"shc"` adds `shc_fermi`
  (`shc_freq_scan = false`) or `shc_freq` (`shc_freq_scan = true`), and `shc` stays `None`. Any other task returns the
  keys above, all `None`. `kubo` is always `None`.
- `gyrotropic_main`: the keys `gyro_K_orb`, `gyro_K_spn`, `gyro_DOS`, `gyro_C`, `gyro_D`, `gyro_Dw`, `gyro_NOA_orb`,
  `gyro_NOA_spn`, all `None` except: if `cwi.win.eval_K` or `cwi.win.eval_spn`, both `gyro_K_orb` and `gyro_K_spn` are
  arrays of shape (3, 3, n_F); `gyro_K_orb` is computed if `eval_K` (else zeros), `gyro_K_spn` if `eval_spn` (else
  zeros). The flags
  `eval_D`, `eval_Dw`, `eval_C`, `eval_NOA` only print a line (not implemented).
- `spin_moment_main`: `Ms_x`, `Ms_y`, `Ms_z`, Python floats, spin magnetic moment in μ_B per cell; all `0.0` if
  `spinors = false`.

**Shapes and units** (n_F = `num_fermi`, n_ω = number of frequencies of `kubo_frequencies`):

| key | shape, type | unit | content |
| --- | --- | --- | --- |
| `ahc_list` | (n_F, 3, 3) float | S/cm | [Fermi energy, term J0/J1/J2, component]; components (x, y, z) = (yz, zx, xy) |
| `morb`, `morb_LC`, `morb_IC` | (n_F, 3, 3) float | μ_B per cell | as `ahc_list`; total, local circulation, itinerant circulation |
| `kubo_H`, `kubo_AH` | (n_ω, 3, 3) complex | S/cm | Hermitian and anti-Hermitian parts of σ_ab(ω), section 5.3 |
| `sc_freq` | (n_ω,) float | eV | frequencies |
| `sc` | (3, 6, n_ω) float | A/V² | σ^{a,bc}(0; ω, −ω); bc = xx, yy, zz, xy, xz, yz |
| `shc_fermi` | (n_F,) float | (ħ/e) S/cm | σ^γ_{αβ} at ω = 0 for each Fermi energy |
| `shc_freq` | (n_ω,) complex | (ħ/e) S/cm | σ^γ_{αβ}(ω) at the single Fermi energy |
| `gyro_K_orb`, `gyro_K_spn` | (3, 3, n_F) float | A (ampere) | K_ab for each Fermi energy: **[a, b, Fermi energy]** |

The terms J0, J1, J2 of `ahc_list` and `morb*` are those of section 5.1. Their split depends on the gauge (`tb_gauge`),
their sum does not; the reference values fix the split in both gauges.

**Frequencies** `kubo_frequencies(cwi)` (used by Kubo, shift current and the ac SHC): with x = (`kubo_freq_max` −
`kubo_freq_min`) / `kubo_freq_step`, n_ω = max(nint(x) + 1, 2), where nint rounds half away from zero (not Python's
`round`); the frequencies are n_ω evenly spaced values from `kubo_freq_min` to `kubo_freq_max`, both included (so the
spacing may differ from the step, and max ≤ min is allowed). A step that is not finite or not positive raises
`SymCWInputError`. Returns a float array (eV).

**Errors** (all `SymCWInputError` from `util/exceptions.py`):

| condition | where |
| --- | --- |
| `num_fermi == 0` | `berry_main`, `gyrotropic_main` |
| `num_fermi != 1` (uses `fermi_energy_list[0]`, never `cwi["fermi_energy"]`) | Kubo, shift current, SHC with `shc_freq_scan`, spin moment |
| `transl_inv = true` | morb, K tensor |
| `spin_decomp = true` | Kubo |
| `zeeman_interaction = true` | shift current |
| `sc_eta < 0` | shift current |
| fixed smearing width ≤ 0 (`kubo_smr_fixed_en_width` when `kubo_adpt_smr = false`; `gyrotropic_smr_fixed_en_width`) | Kubo, shift current, K tensor (not the SHC, which does not check its width) |
| unknown smearing type (section 6) | Kubo, shift current, K tensor (the SHC does not use `kubo_smr_type`) |
| `kubo_freq_step` not finite or ≤ 0 | `kubo_frequencies` |

**Zeeman term** [B]: if `zeeman_interaction`, H(k) is replaced by H(k) + H_Z(k), with H_Z from
`util.utility.spin_zeeman_interaction(magnetic_field, magnetic_field_theta, magnetic_field_phi, σ(k), g_factor,
num_wann)` and σ(k) the Fourier transform of `SS_R`, before the diagonalization, for AHC, morb, Kubo, the K tensor and
the spin moment. H_Z(k) is evaluated from σ(k) at each k point, but it is not differentiated: all ∂_a H used by the formulas come
from `HH_R` alone. It is not
applied for the SHC, and it is an error for the shift current.

**Energy cutoff** [B] for Kubo, shift current and SHC: E_max = `kubo_eigval_max` if it is below 100000; otherwise
`dis_froz_max` + 0.6667 if `dis_froz_max` is present and below 100000; otherwise 100000. Pairs (n, m) with E_n or E_m
above E_max are skipped.

### 2.2 Input keys (`cwi`)

Common: `berry_task` (`berry_main`), `num_wann`, `irvec`, `ndegen`, `unit_cell_cart` (lattice vectors as rows, Å),
`unit_cell_volume` (Å³, positive), `B` (reciprocal lattice vectors as rows, Å⁻¹), `fermi_energy_list`, `num_fermi`,
`tb_gauge` and the positions given by `util.utility.tb_gauge_positions(cwi)`, `transl_inv`, `use_tb_approximation`,
`spinors`, `zeeman_interaction`, `magnetic_field`, `magnetic_field_theta`, `magnetic_field_phi`, `g_factor`.
`cwi.win`: `eval_K`, `eval_spn`, `eval_D`, `eval_Dw`, `eval_C`, `eval_NOA`, `gyrotropic_box`,
`gyrotropic_box_corner`. These values are trusted (checked by the input readers).

| quantity | keys |
| --- | --- |
| AHC | `berry_kmesh`, `berry_curv_unit` (`"ang2"`/`"bohr2"`, only for the threshold), `berry_curv_adpt_kmesh` (3 integers), `berry_curv_adpt_kmesh_thresh` |
| morb | `berry_kmesh` |
| Kubo | `berry_kmesh`, `kubo_freq_min`, `kubo_freq_max`, `kubo_freq_step`, `kubo_eigval_max`, `dis_froz_max` (optional), `kubo_smr_type`, `kubo_smr_fixed_en_width`, `kubo_adpt_smr`, `kubo_adpt_smr_fac`, `kubo_adpt_smr_max`, `spin_decomp`, `use_degen_pert`, `degen_thr` |
| shift current | as Kubo, plus `sc_eta`, `sc_w_thr`, `sc_use_eta_corr` |
| SHC | as Kubo, plus `shc_freq_scan`, `shc_alpha`, `shc_beta`, `shc_gamma` (1, 2, 3 = x, y, z), `shc_bandshift`, `shc_bandshift_firstband` (1-based), `shc_bandshift_energyshift`, and for the Fermi-energy scan `berry_curv_unit`, `berry_curv_adpt_kmesh`, `berry_curv_adpt_kmesh_thresh` (adaptive refinement as in 5.1, with the spin Berry curvature) |
| K tensor | `gyrotropic_kmesh`, `gyrotropic_smr_type`, `gyrotropic_smr_fixed_en_width`, `gyrotropic_smr_max_arg`, `gyrotropic_degen_thresh`, `gyrotropic_band_list` (`None` = all bands, or a string of comma-separated 1-based band numbers), `use_degen_pert`, `degen_thr`, `cwi.win.gyrotropic_box`, `cwi.win.gyrotropic_box_corner` |
| spin moment | `kmesh` |

The meaning of each key is that of the wannier90 user guide (postw90 parameters of the same name) and of
`util/header.py`.

### 2.3 Operators

Real-space matrices in the wannier90 convention O_mn(R) = ⟨m, 0|O|n, R⟩, not divided by `ndegen` (shapes: `HH_R`
(nR, W, W); vectors (3, nR, W, W); tensors (3, 3, nR, W, W)):

- `HH_R`: H. `AA_R`: r_a. `BB_R`: BB_a(k) = i⟨u_m|H|∂_a u_n⟩. `CC_R`: CC_ab(k) = ⟨∂_a u_m|H|∂_b u_n⟩.
- `SS_R` (3, nR, W, W): Pauli matrices σ_γ (dimensionless). `SR_R` (3, 3, nR, W, W), `SH_R` (3, nR, W, W) and `SHR_R`
  (3, 3, nR, W, W): the matrix elements of σ_γ r_a, σ_γ H and σ_γ H r_a of QZYZ18 Eqs. (26)–(27), as defined in the
  docstring of `get_oper_R.get_SHC_R` (axis order [γ, a, R, m, n] and [γ, R, m, n]).

Operators needed: AHC `HH_R`, `AA_R`; morb `HH_R`, `AA_R`, `BB_R`, `CC_R`; Kubo `HH_R`, `AA_R` (or `v_R` with
`use_tb_approximation`); shift current `HH_R`, `AA_R`; SHC `HH_R`, `AA_R`, `SS_R`, `SR_R`, `SH_R`, `SHR_R`; K tensor
`HH_R`, `AA_R`, `BB_R`, `CC_R`, `SS_R`; spin moment `HH_R`, `SS_R`. `SS_R` is also needed with `zeeman_interaction`.
- `v_R`: velocity of the tight-binding approximation (`use_tb_approximation`), from `get_oper_R.get_v_R`.

With `tb_gauge`, the operators are already converted by `get_oper_R.to_tb_gauge`, and the Fourier transforms take the
positions `tb_gauge_positions(cwi)` (orbital positions in the phase).

### 2.4 Helpers that may be used

`util.utility`: `fourier_transform_r_to_k`, `fourier_transform_r_to_k_vec`, `fourier_transform_r_to_k_new` (returns
O(k) and its Cartesian k derivatives), `tb_gauge_positions`, `fermi`, `thermal_avg`, `spn_operator`,
`spin_zeeman_interaction`; `util.get_oper_R.shc_bandshift(E, cwi)`; `util.constants` (SI constants, `hbar_SI`,
`elem_charge_SI`, `elec_mass_SI`, `eV_au`, `bohr`, `bohr_magn_SI`, `joul_to_eV`, `eV_seconds`);
`util.exceptions.SymCWInputError`. These are SymClosestWannier's own code (Fourier transforms, constants, exceptions);
read their docstrings.

## 3. Conventions

- k points in fractional coordinates of the reciprocal lattice; the mesh N1×N2×N3 is k = (i/N1, j/N2, l/N3) with
  0 ≤ i < N1 etc. Integrals over the Brillouin zone: ∫ d³k/(2π)³ F(k) → (1/(N_k V_c)) Σ_k F(k).
- Wannier gauge (W): matrices O^W(k) from the Fourier transform. Hamiltonian gauge (H): O^H = U† O^W U with
  H^W(k) U = U diag(E), eigenvalues in ascending order.
- Occupations at T = 0: f_n = 1 if E_n < E_F, else 0 (strict, as `util.utility.fermi(E − E_F, T=0.0)`; one value per
  Fermi energy).
- Antisymmetric tensors as pseudovectors: component c of X_ab is X_{α(c) β(c)} with (α, β) = (y, z), (z, x), (x, y).
- Cartesian derivatives ∂_a = ∂/∂k_a with k in Å⁻¹.

## 4. Building blocks (WYSV06, YWVS07)

- D^H_a,nm = (U† ∂_a H^W U)_nm / (E_m − E_n) for |E_m − E_n| > 10⁻⁷ eV, 0 otherwise [WYSV06 Eq. (24); threshold B]. In all trace formulas
  write the expressions so that D^H only connects an occupied and an empty state; then degenerate pairs within the
  occupied or the empty manifold never enter.
- Berry connection in the H gauge: A^H_a = Ā^H_a + i D^H_a with Ā^H_a = U† A^W_a U [WYSV06 Eq. (25)].
- Band velocities ∂_a E_n = (U† ∂_a H^W U)_nn [YWVS07 Eq. (27)]. With `use_degen_pert` [B, YWVS07 Eq. (31)]: for each
  direction a separately, group the bands into runs in which consecutive eigenvalues differ by less than `degen_thr`;
  within a run of two or more bands, the band velocities along a are the eigenvalues of the block of U† ∂_a H^W U on
  that run, in ascending order, assigned to the bands of the run in ascending order of energy. Only the band
  velocities change (they enter the adaptive smearing, section 6, and the K tensor); D^H and the matrix elements do not.
- Velocity matrix elements: ħ v_a,nm = (U† ∂_a H^W U)_nm + i (E_n − E_m) Ā^H_a,nm (YWVS07 Sec. II). In the
  tight-binding approximation (`use_tb_approximation`) the velocity is U† v^W U with v^W from `v_R`.

## 5. Quantities

### 5.1 Anomalous Hall conductivity (WYSV06; LVTS12 Eq. (51))

Ω_ab(k) = J0 + J1 + J2 with
- J0 = Σ_n f_n Ω̄^H_ab,nn, Ω̄^W_ab = ∂_a A^W_b − ∂_b A^W_a,
- J1 = Σ_nm (f_m − f_n)(D^H_a,nm Ā^H_b,mn − D^H_b,nm Ā^H_a,mn),
- J2 = i Σ_nm (f_m − f_n) D^H_a,nm D^H_b,mn
[WYSV06 Eq. (32)]; LVTS12 Eq. (51) gives the same as traces with the occupation projector, which avoids D between
occupied states. σ_ab = −(e²/ħ) ∫ d³k/(2π)³ Ω_ab, in S/cm (factor 10⁸ from Å⁻¹ to cm⁻¹). Store each term separately
(`ahc_list[ife, term, c]`).

Adaptive refinement: at a mesh point where |Ω(k)| (Euclidean norm of the pseudovector of the three components, summed over the terms; in bohr²
if `berry_curv_unit = "bohr2"`) exceeds `berry_curv_adpt_kmesh_thresh`, replace the point by a submesh of
ka1×ka2×ka3 points centred on it, at the offsets (m_i + 1/2 − ka_i/2)/(N_i ka_i), m_i = 0, …, ka_i − 1, in fractional
coordinates, each with weight 1/(N_k ka1 ka2 ka3) (no refinement if all ka_i = 1); otherwise use the point
with weight 1/N_k. This is decided for each Fermi energy separately.

### 5.2 Orbital magnetization (LVTS12, CTVR06)

M = (1/N_k) Σ_k [LC(k) + IC(k)] × (conversion to μ_B per cell), LC = (−2 Im g) − E_F (−2 Im f), IC = (−2 Im h) −
E_F (−2 Im f), with −2 Im f the Berry curvature of 5.1, and g, h of LVTS12 Eqs. (56) and (66) written with BB and CC
(the Λ term of LVTS12 Eq. (37) comes from CC). Each of f, g, h splits into J0, J1, J2 terms as in 5.1 (store them
in axis 1). The conversion factor: the result in eV Å² per cell times −`eV_au`/`bohr`² gives μ_B [B]. `transl_inv = true` raises `SymCWInputError`.

### 5.3 Optical conductivity (YWVS07)

YWVS07 Eqs. (1)–(2) split σ_ab(ħω) into a Hermitian (absorptive) part and an anti-Hermitian (reactive) part. They are
computed by two different expressions, which are not the two parts of one complex function when the smearing is not
Lorentzian [B]:

- σ^H_ab(ω) = −π C Σ_k Σ_{n≠m} (f_m − f_n)(E_m − E_n) A_a,nm A_b,mn δ_η(E_m − E_n − ħω),
- σ^AH_ab(ω) = i C Σ_k Σ_{n≠m} (f_m − f_n)(E_m − E_n) A_a,nm A_b,mn / (E_m − E_n − ħω − iη),

with A^H the Berry connection of section 4, f at T = 0 for `fermi_energy_list[0]`, δ_η the smearing of section 6 with
argument x = (E_m − E_n − ħω)/η, and η the width of section 6 (fixed or adaptive per pair and k). The constant is
C = 10⁸ e²/(ħ V_c N_k) (SI e, ħ; V_c in Å³; energies in eV), which gives S/cm [B]. Pairs with E_n or E_m above the cutoff
of section 2.1 are skipped.

Tight-binding approximation (`use_tb_approximation`): the same, with A_a,nm replaced by i v_a,nm/(E_m − E_n) built from
v^H = U† v^W U (v^W from `v_R`), and C = 10⁸ ħ e²/(V_c N_k) [B]. `get_oper_R.get_v_R` returns i R H(R)/`hbar_SI`
(eV Å/(J s)), i.e. ∂_a H/ħ with ħ in SI, which is why C contains ħ.

### 5.4 Shift current (IATS18; Lihm 2021)

σ^{abc}(0; ω, −ω) = C Σ_k Σ_{n≠m} (f_n − f_m) Im[r^b_mn r^{c;a}_nm + r^c_mn r^{b;a}_nm] Δ_nm(ω) [IATS18 Eq. (8)],
with the generalized derivative r^{b;a} by Wannier interpolation as in IATS18 Sec. III (sum rule with the second
derivatives of H^W and the first derivatives of A^W, IATS18 Eqs. (21)–(31)). Pairs with |E_n − E_m| < 10⁻⁷ eV or above
the cutoff of section 2.1 are skipped. Units A/V², with C = `eV_seconds` π e³/(4 ħ² V_c N_k) (SI e, ħ; V_c in Å³;
energies in eV) [B]; derive it from IATS18 Eq. (8) and report any difference (rule 2).

Resonance [B]: Δ_nm(ω) = δ_η(E_m − E_n + ħω) + δ_η(E_n − E_m + ħω), i.e. w(x)/η with x = (E_m − E_n + ħω)/η and
x = (E_n − E_m + ħω)/η respectively (the sign matters for the cold kernel); each term only where |x| ≤ `sc_w_thr`. δ_η
uses `kubo_smr_type` and the Kubo width (`kubo_smr_fixed_en_width`, or adaptive with `kubo_adpt_smr*`, section 6).

`sc_eta` (≥ 0) [B]: in the sums over intermediate states p ≠ n, m of the sum rule for r^{b;a} (IATS18 Eq. (30)), the
D^H in those sums uses the principal-value form (U†∂H U)_pq (E_q − E_p)/((E_q − E_p)² + `sc_eta`²); everywhere else
(r_mn = Ā_mn + i D_mn, the explicit D terms and the outer 1/(E_m − E_n)) D^H is the unregularized one of section 4.
`sc_eta = 0` gives the unregularized expression.

`sc_use_eta_corr` adds, for `sc_eta` > 0, the correction for a finite η of Lihm21 Eq. (19) (comment on IATS18); for
`sc_eta` = 0 it has no effect. The result depends on `tb_gauge` at finite η.

### 5.5 Spin Hall conductivity (QZYZ18; RPS19)

Kubo formula of QZYZ18 Eqs. (1)–(3) for the conventional spin current J^γ_α = {s_γ, v_α}/2 (s_γ = (ħ/2) σ_γ), written
in the units of the Wannier interpolation, where the "velocities" are energy derivatives ħv = ∂H/∂k (eV Å) and the spin
is the Pauli matrix σ (dimensionless):

Ω^γ_{αβ,n}(k, ω) = Σ_{m≠n} −2 Im[j^γ_α,nm (ħv_β)_mn] / ((E_n − E_m)² − (ħω + iη)²)   (Å², complex for ω ≠ 0),
j^γ_α = {σ_γ, ħv_α}/2, matrix elements in the H gauge by Wannier interpolation from `SS_R`, `SR_R`, `SH_R`, `SHR_R`,
`AA_R` and `HH_R` [QZYZ18 Eqs. (23)–(27)];

σ^γ_{αβ} = C (1/N_k) Σ_k Σ_n f_n Ω^γ_{αβ,n}, C = 10⁸ e²/(2 ħ V_c) (SI e, ħ; V_c in Å³), in (ħ/e) S/cm [B]. Derive the
factor of 1/2 (from s = ħσ/2) and the sign from QZYZ18 and report any difference with the reference values (rule 2);
state the units of each intermediate quantity in the docstrings.

- Fermi-energy scan (`shc_freq_scan = false`): ω = 0, with 1/((E_n − E_m)² + η²) in Ω, η = `kubo_smr_fixed_en_width`
  (used as given, no check) or the adaptive width per pair without floor [B]; one value per Fermi energy. Adaptive k
  refinement [B]: only if some ka_i > 1; a mesh point is refined if |Σ_n f_n Ω_n| (in bohr² with `berry_curv_unit =
  "bohr2"`) exceeds `berry_curv_adpt_kmesh_thresh` for at least one Fermi energy; then, as in 5.1, the submesh replaces the point
  for the Fermi energies above the threshold, and the point itself contributes with weight 1/N_k for the others.
  (wannier90 and the old module dropped the point for the others; this was corrected on 2026-10-27, before the
  reference values were made.)
- Frequency scan (`shc_freq_scan = true`): the single Fermi energy, ω from `kubo_frequencies`, ħω + iη with η =
  `kubo_smr_fixed_en_width` (no imaginary part if it is 0), or the adaptive width; no k refinement; complex result.
- `shc_bandshift` [B]: right after the diagonalization, the eigenvalues of the bands from `shc_bandshift_firstband`
  (1-based) on are shifted by `shc_bandshift_energyshift` (`get_oper_R.shc_bandshift(E, cwi)`), and the shifted E_n are
  used in everything that follows: the energy denominators, the occupations, the adaptive widths and the matrix
  elements of j that contain E_n (QZYZ18 Eqs. (25)–(27)). The eigenvectors U and the derivatives of H are not changed.
- The Zeeman term is not applied (section 2.1).

### 5.6 K tensor of the kinetic magnetoelectric effect (TAS17, Eq. (3))

K_ab = Σ_n ∫ d³k/(2π)³ (∂_a E_n) m_nb δ(E_n − E_F) [TAS17 Eq. (3)], computed as [B]

- orbital: K^orb_ab = C_orb Σ_k w_k Σ_n (∂_a E_n) m^orb_nb(k) δ_η(E_n − E_F), with
  m^orb_n = 2 Im[⟨∂u_n| × (H − E_n)|∂u_n⟩] = (−2 Im h_n) − (−2 Im g_n), where (−2 Im g_n) and (−2 Im h_n) are the
  quantities of 5.2 evaluated with band n alone occupied (f_n = 1, f_m = 0 otherwise); note the difference h − g here,
  not the sum of 5.2; C_orb = e²/(2 ħ V_c) (SI; V_c in Å³) [B];
- spin: K^spn_ab = C_spn Σ_k w_k Σ_n (∂_a E_n) ⟨u_n|σ_b|u_n⟩ δ_η(E_n − E_F), C_spn = −10²⁰ e ħ/(2 m_e V_c) [B].

Units: ampere, as in the output files; confirm with the reference values. K^orb is computed only with `eval_K`,
K^spn only with `eval_spn` (section 2.1). ∂_a E_n are the band velocities of section 4
(with `use_degen_pert`). δ_η uses `gyrotropic_smr_type` and `gyrotropic_smr_fixed_en_width` with x = (E_n − E_F)/η;
terms with |x| > `gyrotropic_smr_max_arg` are skipped. Degeneracies [B]: band n is skipped at a k point when
E_n − E_{n−1} ≤ `gyrotropic_degen_thresh` or E_{n+1} − E_n ≤ `gyrotropic_degen_thresh` (neighbours that exist). Bands:
`gyrotropic_band_list` (1-based, comma-separated, each field read with `int()`, so surrounding spaces are allowed and a malformed field raises `ValueError`) or all. k points [B]: the mesh points q of `gyrotropic_kmesh` in the
unit cube (as in section 3) are mapped to k = `gyrotropic_box_corner` + q · `gyrotropic_box` (fractional coordinates,
rows of the box are its edge vectors), with weight w_k = det(`gyrotropic_box`)/N_k (signed determinant, no wrapping).

## 6. Smearing

- Types (`*_smr_type`) [B]: `gauss`; `m-pN` with N a decimal integer 0 ≤ N ≤ 10 (Methfessel–Paxton of order N; `m-p0` is
  `gauss`, `m-p` alone is order 1); `m-v` or `cold` (Marzari–Vanderbilt); `f-d` (Fermi–Dirac). Anything else raises
  `SymCWInputError`.
- δ_η(E) = w(E/η)/η with the kernels of the papers: gauss w(x) = e^{−x²}/√π; Methfessel–Paxton
  w(x) = Σ_{n=0}^{N} A_n H_{2n}(x) e^{−x²}, A_n = (−1)^n/(n! 4^n √π) [MP89]; cold
  w(x) = e^{−(x − 1/√2)²}(2 − √2 x)/√π [MV99]; Fermi–Dirac w(x) = 1/(2 + 2 cosh x). The cold kernel is not even: its
  argument x is the one given in each formula (5.3: (E_m − E_n − ħω)/η; 5.6: (E_n − E_F)/η).
- Fixed width: `*_smr_fixed_en_width` (> 0).
- Adaptive width (`kubo_adpt_smr`) [YWVS07 Eqs. (34)–(35)]: for a pair (n, m) at k, η = min(`kubo_adpt_smr_fac` ×
  |∇_k E_m − ∇_k E_n| × Δk (Euclidean norm of the Cartesian band velocities, eV Å), `kubo_adpt_smr_max`), at least 10⁻⁶ eV for Kubo and the shift current (the SHC has no floor) [B], with the band velocities of section 4 and Δk the
  largest of |b_i|/N_i over the reciprocal vectors b_i (rows of `B`) and the mesh `berry_kmesh`.

## 7. Validation

### 7.1 Tests

1. `tests/test_response_reference.py` must pass: 28 cases on the reference model of `tests/reference_model.py` (a
   random 3-band model without symmetry, with exact real-space operators; 4 Fermi energies, so that the Fermi-energy
   axis differs in length from the Cartesian ones). It compares every returned array with
   `tests/data/response_reference.npz` to a relative 1e-7 of the largest value of the array, including the terms
   J0/J1/J2 in both gauges, and the set of keys whose value is `None`. The helper `tests/response_reference.py`
   builds the inputs and calls only the public functions of section 2.1. The cases cover: AHC (both gauges, adaptive k
   refinement in Å² and bohr²), morb (both gauges), Kubo (smearing gauss, m-p1, m-v, f-d; adaptive smearing;
   `kubo_eigval_max` cutting the top band; tb gauge; tight-binding approximation), shift current (with and without the
   η correction; tb gauge), SHC (Fermi scan with two index orders, with adaptive smearing and with adaptive k refinement that is triggered for
   some Fermi energies only; frequency scan with and
   without band shift), the K tensor (gauss, m-p1, f-d) and the spin moment (with and without the Zeeman term).
2. Not covered by the reference values: degenerate bands (`use_degen_pert`, `degen_thr`, `gyrotropic_degen_thresh`; the
   random model has no degeneracies), `spinors = false`, `gyrotropic_band_list`, a nontrivial `gyrotropic_box`, cold
   smearing outside Kubo, the errors of section 2.1. Before the old module is removed, the maintainers add reference
   cases for these (a model with an exact twofold degeneracy, a box, a band list), generated with the old module. With exact degeneracies, compare only quantities that do not depend on the basis of the
   degenerate subspace (totals over bands).
3. Tests that call internal functions of the old module (`test_tb_gauge.py` partly, `test_berry_morb_sc.py`,
   `test_gyrotropic.py`, `test_smearing.py`, `test_kubo_frequencies.py`, `test_shc_bandshift.py`,
   `test_fermi_energy.py`) are not given to the implementer (rule 1). After the implementation, the maintainers port
   them to the public functions, keeping what they check (errors, frequency grid, smearing names, gauge independence of
   the totals, morb = LC + IC, the optional comparisons with wannier-berri, which use it as a black box).
4. Comparison with postw90: there is no automated test in the repository. The agreement of postcw with postw90 was
   checked by hand (Fe example of wannier90: AHC and orbital magnetization); the maintainers repeat it for the new
   implementation. Turning it into a reproducible benchmark with recorded inputs is a follow-up.

### 7.2 Limits of this validation

The reference values come from the implementation being replaced: they check backward compatibility, not the physics.
Independent checks (analytic two-band models, symmetry-forbidden components, mesh convergence, published benchmarks)
are follow-ups, not conditions of the rewrite. The constants marked [B] and the signs of the terms are fixed by the
reference values; the implementer derives them from the papers and reports any disagreement (rule 2).

Behaviour kept for compatibility although it looks like a defect (fixing it is a separate, later change with new
reference values): the SHC accepts any `kubo_smr_type` and a width ≤ 0.

## 8. Papers

| key | reference |
| --- | --- |
| WYSV06 | X. Wang, J. R. Yates, I. Souza, D. Vanderbilt, Phys. Rev. B 74, 195118 (2006) |
| YWVS07 | J. R. Yates, X. Wang, D. Vanderbilt, I. Souza, Phys. Rev. B 75, 195121 (2007) |
| LVTS12 | M. G. Lopez, D. Vanderbilt, T. Thonhauser, I. Souza, Phys. Rev. B 85, 014435 (2012) |
| CTVR06 | D. Ceresoli, T. Thonhauser, D. Vanderbilt, R. Resta, Phys. Rev. B 74, 024408 (2006) |
| IATS18 | J. Ibañez-Azpiroz, S. S. Tsirkin, I. Souza, Phys. Rev. B 97, 245143 (2018) |
| QZYZ18 | J. Qiao, J. Zhou, Z. Yuan, W. Zhao, Phys. Rev. B 98, 214402 (2018) |
| RPS19 | J. H. Ryoo, C.-H. Park, I. Souza, Phys. Rev. B 99, 235113 (2019) |
| TAS17 | S. S. Tsirkin, P. Aguado Puente, I. Souza, Phys. Rev. B 97, 035158 (2018) |
| Lihm21 | J.-M. Lihm, Phys. Rev. B 103, 247101 (2021) (comment on IATS18) |
| MP89 | M. Methfessel, A. T. Paxton, Phys. Rev. B 40, 3616 (1989) |
| MV99 | N. Marzari, D. Vanderbilt, A. De Vita, M. C. Payne, Phys. Rev. Lett. 82, 3296 (1999) |

All are on arXiv; find them by title.
