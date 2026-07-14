#Rotaxane builder

## code to generate an XYZ structure for a rotaxane from SMILES

- Rotaxanes are molecular machines that have a long molecule (we will call it the rod) and a cyclic molecule (we will call it the wheel)
- The rod is threaded through the wheel.

## your task
- you will read in two smiles, rod then wheel
- generate a 3D structure for each using RDKit
- the rod will have one dimension with longer extent than the other two. This is the rod's long axis. Find a coordinate transformation that will align this axis with the x-axis so that most coordinates are largest on the x and smaller on the y and z.
- the wheel will primarily exist in 2 dimensions, which its extent in those dimensions larger than the others. Find a coordinate transformation that will align this plane with the yz plane
- find the centroid of each of the aligned sets of coordinates. translate the wheel so that it's centroid overlaps with the rod's centroid.
- check for coordinate overlap. Since the rod is primarily on the x axis and the wheel is primarily on the yz plane, this should be minimal.

complete these tasks and then output the combined molecule in xyz format for visual inspection. 

## tech details
- the rod and wheel smiles are in a `.txt` file (default `rot_smiles.txt`). the
  format is:
  line 1 --> rod: [rod smiles]
  line 2 --> wheel: [wheel smiles]
  optional extra lines (used only by optimize_uma.py / run_md.py, defaults shown):
  charge: [int]   (default 0)
  spin: [int]     (default 1, spin multiplicity)
- **repository layout.** scripts live in `code/`; the input `<stem>.txt` files
  live at the project root; generated run data (structures `.xyz`, trajectories
  `.pdb`, scan tables `.csv`, logs `.log`) lives in `output_files/`; plots
  (`.png`) and figures live in `images/`. the scripts auto-route outputs by
  extension, so a bare run writes to the right folder with no `--out-*` flags.
- **stem-driven filenames.** every output file is named from the input `.txt`
  file's stem (basename without `.txt`): `rot1.txt` -> `output_files/rot1_center.xyz`,
  `output_files/rot1_relaxed.xyz`, `output_files/rot1_displaced.xyz`,
  `output_files/rot1_md.*`, etc. downstream stages recover the stem by stripping
  the role suffix (`_center`, `_relaxed`, `_displaced`, `_displaced_relaxed`,
  `_md`, `_scan`) from whatever `.xyz` they read, then name their outputs
  `<stem>_<role>.<ext>`. helpers live in `code/rotaxane_paths.py` (`resolve_stem`,
  `out_path`, `default_smiles`, plus the `OUTPUT_DIR` / `IMAGES_DIR` /
  `PROJECT_ROOT` constants it derives from its own `code/` location). `out_path`
  defaults to `images/` for `.png` and `output_files/` for everything else;
  `default_smiles` defaults to the project root. all `--out-*` flags still exist
  to override. the default stem is `rot_smiles` (from `rot_smiles.txt`), so the
  old `rotaxane*` outputs are legacy.
- **engine (force source).** optimize_uma / run_md / displace_wheel all take
  `--engine uma|tblite` (default `uma`) + `--method` (tblite xTB method, default
  `GFN2-xTB`; also GFN1-xTB / GFN0-xTB / CEH; ignored for uma). `uma` = Meta UMA
  MLIP (`uma-s-1p1`, `omol`) via fairchem (needs `HF_TOKEN`, ~1.07 s/relax-step on
  CPU). `tblite` = GFN-xTB tight-binding via the tblite ASE calculator (NO
  `HF_TOKEN`, ~0.23 s/step, ~5x faster). the engine is chosen once per run and
  the calculator is built by `make_calculator` in `optimize_uma.py`, which
  imports the engine's libs *inside* the branch it uses -- so a single process
  loads only one engine. this is structural, not stylistic: torch/fairchem and
  tblite each bundle their own `libomp` and segfault (exit 139) if used in one
  process; `KMP_DUPLICATE_LIB_OK` does not fix it. that is why the old top-level
  `import torch` / `from fairchem.core import ...` moved OUT of module scope
  (run_md / displace_wheel `from optimize_uma import ...`, so a module-level
  torch import would pull torch into tblite runs too). **naming:** a non-default
  engine tags every output, UMA keeps the legacy untagged names for back-compat:
  `rot2_relaxed.xyz` (uma) vs `rot2_relaxed_tblite.xyz` (tblite); likewise
  `<stem>_md_tblite.*`, `<stem>_scan_tblite.csv`/`.png`, `<stem>_displaced_tblite.xyz`,
  and the `--dump-stations` dir `<stem>_stations_tblite/` (uma = `<stem>_stations/`).
  `resolve_stem` strips the trailing engine tag *then* the role suffix, so the
  pipeline chains on either engine end-to-end (`rot2_relaxed_tblite.xyz` -> stem
  `rot2` -> `rot2_displaced_tblite.xyz`).
- single project environment is `.venv` (Python 3.12, created with `uv`).
  it has `rdkit`, `fairchem-core` (brings `ase` + `torch`), installed via
  `uv pip install --python .venv/bin/python rdkit fairchem-core`.
  note: fairchem-core requires Python <3.14, which is why the env is 3.12
  and not the system 3.14. run scripts from the project root as
  `.venv/bin/python code/<script>` (so `code/` is on sys.path for the
  cross-script imports like `from optimize_uma import ...`).
- pipeline (stage 3 optional):
  1. `code/build_rotaxane.py` -- RDKit 3D embed + PCA alignment + centroid/threading
     sterics optimization -> writes `output_files/<stem>_center.xyz` (default stem
     `rot_smiles`). CLI: `--smiles` (input .txt, sets the stem), `--out`.
  2. `code/optimize_uma.py` -- relaxes with Meta's UMA MLIP (`uma-s-1p1`, `omol`
     task) via the fairchem ASE calculator. CLI: `--input`, `--out-xyz`,
     `--out-pdb`, `--fmax`, `--steps`, `--smiles` (defaults relax
     `output_files/<stem>_center.xyz` -> `output_files/<stem>_relaxed.xyz` +
     `output_files/<stem>_relax.pdb`). writes
     PyMOL-friendly outputs only: plain standard XYZ of the final frame and a
     multi-state PDB of the whole relaxation (`mplay` in PyMOL). plain XYZ is
     written by hand because ASE's xyz writer emits extended XYZ (forces + long
     Lattice/Properties comment) that PyMOL misreads. needs `HF_TOKEN` in the
     environment (huggingface access to the UMA weights) to download the
     checkpoint -- UMA only; `--engine tblite` needs no token. device
     auto-selects cuda if available else cpu (fairchem only accepts cpu/cuda --
     Apple Silicon MPS is not supported; tblite has no torch device). charge
     and spin are read from optional `charge:`/`spin:` lines in `<stem>.txt`,
     defaulting to 0 and 1. CLI also: `--engine uma|tblite` (default uma),
     `--method` (tblite GFN2-xTB).
  3. `code/displace_wheel.py` -- maps the shuttle landscape and places the MD
     start. does two things: (a) a **chain-seeded relaxed stability scan** -- two
     monotonic outward sweeps from the central relaxed minimum, each station
     seeded from the previous relaxed geometry so the wheel threads through a
     stopper incrementally (a fresh rigid start past a stopper sits in deep
     overlap and does not converge on CPU). each sweep walks until it has mapped
     every well on its side and hits the rod-tip wall (energy/contact cutoff,
     rod-length-aware), so multiple wells on a long rod are captured. at each
     station the wheel is held RIGID (FixAtoms) at its station and the rod's two
     endpoint atoms are anchored, so the only free DOF is the rod's internal
     flex (phenyl/CF3 groups rotating away from the wheel) -- the scan coordinate
     stays fixed while bad sterics are relieved; tight convergence
     (`--scan-fmax 0.05`, `--scan-steps 200`). writes `images/<stem>_scan.png`
     (energy vs displacement in kcal/mol, relative to the global min, with
     global-min and well markers) and `output_files/<stem>_scan.csv` (4-col:
     displacement_A, energy_eV, energy_rel_kcal_mol, min_contact_A). the
     landscape is on the wheel-rigid / rod-endpoint-anchored constrained
     surface, so it includes some rod-conformer relaxation; well depths/barriers
     are on that surface, not the free PES. and (b) detects the scan's wells
     (local minima separated from the global min by > BARRIER_MIN=3.0 kcal/mol,
     after smoothing rod-conformer bumps; each basin is snapped to its deepest
     converged raw point so the 1.25 A smoothing window doesn't shift the
     reported minimum off the true floor; NOT-converged stations are interpolated
     from converged neighbours and excluded from the global min) and writes the
     chosen well's scan-relaxed geometry to `output_files/<stem>_displaced.xyz`.
     well depths are grid-resolution-limited (0.25 A default vs the ~0.1 A that
     resolves a ~0.6 A well) -- use `--scan-grid 0.1` for publication-quality
     depths (slower). needs
     HF_TOKEN (UMA). rod/wheel counts come from `<stem>.txt` via RDKit (must
     match build). a stopper is detected as a *sustained* rod-wheel overlap (min
     distance < 1.0 A over a 3 A width), not transient dips from bumpy rod
     features; the search is rod-length-aware (bounded by the rod's extent, not
     a fixed cap) and is used only for the advisory plot annotation and the
     rigid fallback. CLI: `--side left|right|farther|deeper` (default farther =
     largest |d| well; deeper = lowest-energy well), `--place-rigid` (force the
     legacy rigid stopper-wall placement, which is strained and must be relaxed
     next), `--input` (default `output_files/<stem>_relaxed.xyz`), `--out`,
     `--scan-grid` (A, default 0.25), `--scan-fmax` (eV/A, default 0.05),
     `--scan-steps` (default 200), `--production` (raise --scan-steps to a floor
     of 300 so the hard threading-event stations converge instead of hitting the
     step cap with an unreliable energy; for reporting-quality well depths --
     only the 1-3 threading points use the extra steps, the rest converge <100),
     `--scan-pad` (A, default 4.0; rigid scan only),
     `--scan-walk-emax` (kcal/mol above min to stop a sweep, default 40),
     `--scan-walk-contact` (A; stop when min_contact drops below, default 1.2),
     `--scan-emax` (kcal/mol plot clip, default none), `--rate-temp` (K, default
     300; temperature for the Eyring rate-estimate block), `--no-rates` (skip the
     rate-estimate block), `--no-scan`,
     `--no-scan-chain` (legacy rigid fresh-start scan; cannot map past a
     stopper), `--engine uma|tblite` (default uma; tblite tags scan/displaced/
     stations outputs `_tblite`), `--method` (tblite GFN2-xTB). the scan summary prints an Eyring TST rate estimate per well --
     escape (well -> global min, over the well's barrier) and entry (global min
     -> well, over the same saddle, = rel_kcal + barrier) as k and tau at
     --rate-temp; ESTIMATE only (constrained-surface potential barriers treated
     as free-energy barriers with the kBT/h prefactor; true rates need a
     free-energy profile / Kramers friction). stations that hit the step cap
     (NOT-converged, typically a wheel mid-threading) report an unreliable energy
     and are excluded from well detection -- interpolated from converged
     neighbours before smoothing, barred
     from being a well/global-min, and the global min is taken over converged
     stations only; the CSV/plot relative-energy scale uses that converged-only
     min. the well geometry is already scan-relaxed and MD-ready, so the
     displaced-relax step is now OPTIONAL -- `run_md.py` runs on it directly.
     with `--place-rigid`, relax first:
     `code/optimize_uma.py --input output_files/<stem>_displaced.xyz` (auto-names
     `output_files/<stem>_displaced_relaxed.xyz` + `output_files/<stem>_displaced_relax.pdb`).
  4. `code/run_md.py` -- MD using UMA for forces. CLI: `--dt` (fs, default 0.5),
     `--time` (fs, default 100, short test), `--thermostat` (langevin NVT
     default | nve), `--temperature` (K, default 300), `--friction` (1/fs,
     default 0.01), `--stride`, `--seed`, `--input` (default
     `output_files/<stem>_relaxed.xyz`), `--out-xyz`, `--out-pdb` (defaults
     `output_files/<stem>_md.xyz` / `output_files/<stem>_md.pdb`), `--smiles`,
     `--engine uma|tblite` (default uma; tblite tags md outputs `_tblite`),
     `--method` (tblite GFN2-xTB). reuses helpers from
     optimize_uma.py. same HF_TOKEN (uma only) / device / charge-spin rules.
  5. `code/plot_md.py` -- plots MD observables from a run_md.py stdout log + the
     multi-state PDB. CLI: `--log`, `--pdb` (default `output_files/<stem>_md.pdb`, also sets
     the stem), `--prefix` (default `images/<stem>_md_`), `--dt`, `--log-interval`,
     `--no-rmsd`. writes `<prefix>temperature|energy|wheel|rmsd|overview.png`.
