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
     checkpoint. device auto-selects cuda if available else cpu (fairchem
     only accepts cpu/cuda -- Apple Silicon MPS is not supported). charge
     and spin are read from optional `charge:`/`spin:` lines in `<stem>.txt`,
     defaulting to 0 and 1.
  3. `code/displace_wheel.py` (optional) -- slides the wheel along the rod's PCA
     long axis. does two things: (a) a **relaxed stability-vs-position scan** --
     the wheel is slid across the clash-free travel window on a grid; at each
     station the wheel is pinned along the rod axis (FixedPlane) and the rod is
     held rigid (FixAtoms) while UMA relaxes the wheel in the perpendicular
     plane with a LOOSE convergence (just enough to relieve bad sterics, not a
     full minimisation), then the relaxed energy is recorded. this gives a
     finite shuttle landscape with real minima/barriers as the wheel passes over
     the rod's phenyl/CF3 features. writes `images/<stem>_scan.png` (energy vs
     displacement, relative to min, stopper walls and placed extreme marked) and
     `output_files/<stem>_scan.csv`; and (b) places the wheel at a steric extreme for a more
     interesting MD start -> `output_files/<stem>_displaced.xyz`. needs HF_TOKEN (UMA).
     rod/wheel counts come from `<stem>.txt` via RDKit (must match build). a
     stopper is detected as a *sustained* rod-wheel overlap (min distance < 1.0 A
     over a 3 A width), not the transient dips from bumpy rod features passing
     through the ring. CLI: `--side left|right|farther` (default farther),
     `--margin`, `--input` (default `output_files/<stem>_relaxed.xyz`), `--out`,
     `--scan-grid` (A, default 0.5), `--scan-pad` (A, default 0),
     `--scan-fmax` (eV/A, default 0.5), `--scan-steps` (default 20),
     `--scan-emax` (eV plot clip, default none), `--no-scan`. the rigid
     displacement is strained, so relax it next:
     `code/optimize_uma.py --input output_files/<stem>_displaced.xyz` (auto-names
     `output_files/<stem>_displaced_relaxed.xyz` + `output_files/<stem>_displaced_relax.pdb`).
  4. `code/run_md.py` -- MD using UMA for forces. CLI: `--dt` (fs, default 0.5),
     `--time` (fs, default 100, short test), `--thermostat` (langevin NVT
     default | nve), `--temperature` (K, default 300), `--friction` (1/fs,
     default 0.01), `--stride`, `--seed`, `--input` (default
     `output_files/<stem>_relaxed.xyz`), `--out-xyz`, `--out-pdb` (defaults
     `output_files/<stem>_md.xyz` / `output_files/<stem>_md.pdb`), `--smiles`. reuses helpers from
     optimize_uma.py. same HF_TOKEN / device / charge-spin rules.
  5. `code/plot_md.py` -- plots MD observables from a run_md.py stdout log + the
     multi-state PDB. CLI: `--log`, `--pdb` (default `output_files/<stem>_md.pdb`, also sets
     the stem), `--prefix` (default `images/<stem>_md_`), `--dt`, `--log-interval`,
     `--no-rmsd`. writes `<prefix>temperature|energy|wheel|rmsd|overview.png`.
