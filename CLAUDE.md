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
- the rod and wheel smiles are in a file called rot_smiles.txt. the format is:
  line 1 --> rod: [rod smiles]
  line 2 --> wheel: [wheel smiles]
  optional extra lines (used only by optimize_uma.py, defaults shown):
  charge: [int]   (default 0)
  spin: [int]     (default 1, spin multiplicity)
- single project environment is `.venv` (Python 3.12, created with `uv`).
  it has `rdkit`, `fairchem-core` (brings `ase` + `torch`), installed via
  `uv pip install --python .venv/bin/python rdkit fairchem-core`.
  note: fairchem-core requires Python <3.14, which is why the env is 3.12
  and not the system 3.14. run scripts with `.venv/bin/python <script>`.
- pipeline (stage 3 optional):
  1. `build_rotaxane.py` -- RDKit 3D embed + PCA alignment + centroid/threading
     sterics optimization -> writes `rotaxane.xyz`.
  2. `optimize_uma.py` -- relaxes with Meta's UMA MLIP (`uma-s-1p1`, `omol`
     task) via the fairchem ASE calculator. CLI: `--input`, `--out-xyz`,
     `--out-pdb`, `--fmax`, `--steps` (defaults relax `rotaxane.xyz` ->
     `rotaxane_uma_clean.xyz` + `rotaxane_uma.pdb`). writes PyMOL-friendly
     outputs only: plain standard XYZ of the final frame and a multi-state
     PDB of the whole relaxation (`mplay` in PyMOL). plain XYZ is written by
     hand because ASE's xyz writer emits extended XYZ (forces + long
     Lattice/Properties comment) that PyMOL misreads. needs `HF_TOKEN` in the
     environment (huggingface access to the UMA weights) to download the
     checkpoint. device auto-selects cuda if available else cpu (fairchem
     only accepts cpu/cuda -- Apple Silicon MPS is not supported). charge
     and spin are read from optional `charge:`/`spin:` lines in
     rot_smiles.txt, defaulting to 0 and 1.
  3. `displace_wheel.py` (optional) -- slides the wheel along the rod's PCA
     long axis to a steric extreme for a more interesting MD start. rod/wheel
     counts come from rot_smiles.txt via RDKit (must match build). a stopper is
     detected as a *sustained* rod-wheel overlap (min distance < 1.0 A over a
     3 A width), not the transient dips from bumpy rod features passing through
     the ring. CLI: `--side left|right|farther` (default farther), `--margin`,
     `--input` (default `rotaxane_uma_clean.xyz`) -> `rotaxane_displaced.xyz`.
     the rigid displacement is strained, so relax it next:
     `optimize_uma.py --input rotaxane_displaced.xyz --out-xyz
     rotaxane_displaced_relaxed.xyz --out-pdb rotaxane_displaced_relax.pdb`.
  4. `run_md.py` -- MD using UMA for forces. CLI: `--dt` (fs, default 0.5),
     `--time` (fs, default 100, short test), `--thermostat` (langevin NVT
     default | nve), `--temperature` (K, default 300), `--friction` (1/fs,
     default 0.01), `--stride`, `--seed`, `--input` (default
     `rotaxane_uma_clean.xyz`), `--out-xyz`, `--out-pdb` (defaults
     `rotaxane_md_clean.xyz` / `rotaxane_md.pdb`). reuses helpers from
     optimize_uma.py. same HF_TOKEN / device / charge-spin rules.
