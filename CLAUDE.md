# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

SymClosestWannier (SymCW) builds closest Wannier (CW) tight-binding models from plane-wave DFT outputs (wannier90-style
`seedname.win/amn/eig/nnkp`). Optionally, it symmetrizes them with the Symmetry-Adapted Multipole Basis (SAMB) from
[MultiPie](https://github.com/CMT-MU/MultiPie). It requires Python >= 3.12.

## Branch workflow

- The default branch is `main`. Every PR branches from `develop` and targets `develop` (`gh pr create --base develop`).
- After a merge into `develop`, delete the feature branch (`gh pr merge --delete-branch`), pull `develop`, and run `pytest`.
- The maintainer opens and merges the `develop` → `main` PR by hand. Never open or merge it, and never delete `develop`.

## Commands

```bash
pip install -e ".[dev]"                     # dev extras: pytest, black, jupyter-book<2, ghp-import, multipie, qtdraw
pytest -n auto                              # full suite (~20 s; -n needs pytest-xdist, not in the dev extras)
pytest tests/test_pw2cw.py::test_name       # single test
black --line-length=130 .                   # formatter (line length 130, from pyproject.toml)
make -C docs build                          # Jupyter Book docs -> docs/_build/html (CI publishes from main)
pw2cw SEEDNAME                              # build the CW model (reads SEEDNAME.cwin in the current dir)
postcw SEEDNAME                             # analyze it (reads SEEDNAME.hdf5 written by pw2cw)
pw2cw -i                                    # list the .cwin input keywords
```

Some tests compare with wannier-berri and are skipped unless it is installed (`pip install "wannierberri>=26.7" "numba>=0.60"`).
CI runs a separate job with wannier-berri installed, and that job fails if any test is skipped.

## Architecture

There are two CLI entry points (`symclosestwannier/scripts/`, both built with click and `scripts/common.py:run_command`):

- **`pw2cw`** → `cw/cw_creator.py:cw_creator`. It reads inputs from the current directory, then `CWManager` `chdir`s into
  `outdir`, and the original directory is restored in a `finally` block. Flow: `CWin` (the `.cwin` keys) → `CWInfo`
  (aggregates `Win`, `Nnkp`, `Eig`, `Amn`, plus optional `Mmn`, `Umat`, `Spn`, `UHu` readers in `cw/`) → `CWModel` → writes
  `seedname.cwout`, the `*.dat.cw` files and `seedname.hdf5`.
- **`postcw`** → `analyzer/analyzer.py:analyzer`. It reloads the model from `seedname.hdf5`
  (`CWModel.read_info_data`), merges in the `.cwin` and `.win` settings, and runs `Response` (`analyzer/response.py`, whose
  numerics are in `analyzer/get_response.py`: AHC, orbital magnetization, Kubo, shift current, SHC, gyrotropic K, spin
  moment) and `Band`. Output goes to `seedname.cwpout`.

`CWInfo`, `CWModel` and `CWin` are all `dict` subclasses, so settings and data are accessed by key (`cwi["num_wann"]`).

`CWModel` (`cw/cw_model.py`) is the core:
- `restart = "cw"` → `_cw()`: exclude bands below `proj_min` → disentangle with the window function (`cwf_mu_*`,
  `cwf_sigma_*`, `cwf_delta`) → `_construct_tb` builds `Hk/Sk/nk` and Fourier-transforms them to `Hr/Sr/nr` on the
  Wigner–Seitz `irvec`/`ndegen` (`util/utility.py:wigner_seitz`).
- `restart = "w90"` → `_w90()`: uses wannier90 U matrices instead.
- `symmetrization = true` → `_sym()`: loads a MultiPie `MaterialModel` (`mp_seedname`, `mp_outdir`) in `CWInfo`, projects
  the operators onto the SAMB (`samb_decomp_operator`, `construct_Or`), and stores the `*_sym` matrices.

Real-space operators for responses (`AA_R`, `BB_R`, `CC_R`, `SS_R`, `v_R`, …) come from `util/get_oper_R.py`.

### Conventions that are easy to break

- **Phase convention:** this is the wannier90 convention. `O[iR, a, b]` is not divided by `ndegen`, and the Bloch phase
  has no orbital positions. `tb_gauge = true` only changes what `postcw` passes to the calculation
  (`get_oper_R.py:to_tb_gauge`); stored and written matrices never change. See `docs/src/format/convention.md`.
- **Orbital order:** plain matrices use the `ket_amn` order (the projections in `.win`). `*_sym` matrices use MultiPie's
  SAMB ket order. Convert between the two with `util/utility.py:sort_ket_matrix`.
- **Input keywords:** `CWin._default()` in `cw/cwin.py` defines the accepted keys. A new key must also be added to
  `util/header.py` (`cwin_info`, shown by `pw2cw -i`), to the `CWin.read` docstring, and to `docs/src/format/cwin.md`.
  `tests/test_cwin_keys.py` enforces this.
- **Errors:** errors caused by inputs or settings raise subclasses of `util/exceptions.py:SymCWError`
  (`SymCWInputError`, `SymCWFileNotFoundError`). The CLI shows these without a traceback.
- **Input files:** they are opened through `util/input_file.py:open_input`, which also reads compressed files and tar
  archives.

## Tests and reference data

- `tests/conftest.py:make_case` copies DFT inputs from `others/examples/<seed>/` into a temporary directory, writes a
  `.cwin` from `CWIN`, and `chdir`s into it.
- Regression references live in `tests/data/`; their provenance is in `tests/data/README.md`. Hamiltonians are compared
  with `atol = 1e-7`.
- `tests/response_reference.py` + `tests/data/response_reference.npz` pin the outputs of `postcw` responses on a model
  (`tests/reference_model.py`). Never regenerate the reference with `--write` just to make a change pass. Regenerate it
  only to add or change reference cases.
- `others/response_rewrite/spec.md` is a clean-room specification for re-implementing `analyzer/get_response.py` (ported
  from wannier90 postw90, GPL) from the papers. Whoever writes that rewrite must not read the old module, wannier90's
  source or wannier-berri's source, and must follow the stripping rules in the spec.
