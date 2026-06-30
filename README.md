# Rotaxane builder + UMA molecular dynamics

Build a rotaxane (rod threaded through a wheel) from two SMILES, relax it with
Meta's **UMA** ML potential, and run ab-initio-style MD with UMA forces.

```
SMILES  ─►  build_rotaxane.py  ─►  rotaxane.xyz
                                      │
                                      ▼
                          optimize_uma.py  ─►  rotaxane_uma_clean.xyz  (+ .pdb)
                                      │
                          (optional) displace_wheel.py  ─►  rotaxane_displaced.xyz
                                      │
                          (optional) optimize_uma.py --input ...  ─►  relaxed displaced
                                      │
                                      ▼
                              run_md.py  ─►  rotaxane_md_clean.xyz  (+ .pdb)
```

## Demo: SMILES → 3D

Each rotaxane is specified by just two SMILES strings — a rod and a wheel.
`build_rotaxane.py` turns those strings into a 3D structure: RDKit embeds each
fragment (`ETKDGv3` + `MMFF`), the rod's long axis is PCA-aligned to x and the
wheel's plane to yz, the wheel is threaded onto the rod at the steric-energy
minimum, and the result is then relaxed with UMA. The pictures below are the
relaxed structures rendered from the UMA-optimised XYZ.

### Rotaxane 1 (`rot_smiles.txt`) — 144 atoms (rod 88 + wheel 56)

```
rod:   O=C(C(C(F)=C1)=CC=C1C(C=C2)=CC=C2C(C=C3)=CC=C3C4=CC(F)=C(C(N(CC5=CC(C(F)(F)F)=CC(C(F)(F)F)=C5)[H])=O)C=C4)N(CC6=CC(C(F)(F)F)=CC(C(F)(F)F)=C6)[H]
wheel: O1CCOCCOCCOCCOCCOCCOCCOCC1
```

![Rotaxane 1](Rotaxane1.png)

### Rotaxane 2 (`rot_smiles2.txt`) — 114 atoms (rod 58 + wheel 56)

```
rod:   O=C(C(C(F)=C1)=CC(F)=C1C(N(CC2=CC(C(F)(F)F)=CC(C(F)(F)F)=C2)[H])=O)N(CC3=CC(C(F)(F)F)=CC(C(F)(F)F)=C3)[H]
wheel: O1CCOCCOCCOCCOCCOCCOCCOCC1
```

![Rotaxane 2](Rotaxane2.png)

(The wheel is the same 24-crown-8 in both; only the rod differs.)

## Inputs

`rot_smiles.txt` (two required lines, two optional):

```
rod:   <rod SMILES>
wheel: <wheel SMILES>
charge: 0      # optional, default 0 (used by optimize_uma.py / run_md.py)
spin: 1        # optional, default 1 (spin multiplicity)
```

The rod is a long, roughly 1D molecule; the wheel is a roughly 2D ring
(24-crown-8 here). `build_rotaxane.py` aligns the rod's long axis to x and the
wheel's plane to yz, co-centroids them, then nudges the wheel in yz and slides
it along x to minimise steric clashes.

## Setup

 Requires `uv` (Homebrew: `brew install uv`) and a HuggingFace token with
access to the UMA weights.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python rdkit fairchem-core
export HF_TOKEN=<your huggingface token>   # put in ~/.zshrc
```

Notes:
- `fairchem-core` requires Python **<3.14**, so the env is 3.12 (not the
  system 3.14). Run scripts with `.venv/bin/python <script>`.
- `fairchem`'s MLIP predict unit only accepts `cpu` or `cuda` — Apple Silicon
  **MPS is not supported**. Device auto-selects `cuda` if available else `cpu`.

## Usage

```sh
# 1. assemble the rotaxane from SMILES (RDKit) -> rotaxane.xyz
.venv/bin/python build_rotaxane.py

# 2. relax with UMA -> rotaxane_uma_clean.xyz + rotaxane_uma.pdb
.venv/bin/python optimize_uma.py

# 3. (optional) slide the wheel to a steric extreme for shuttling MD
.venv/bin/python displace_wheel.py            # --side left|right|farther

# 3b. (optional) relax the displaced structure
.venv/bin/python optimize_uma.py \
    --input rotaxane_displaced.xyz \
    --out-xyz rotaxane_displaced_relaxed.xyz \
    --out-pdb rotaxane_displaced_relax.pdb

# 4. MD with UMA forces -> rotaxane_md_clean.xyz + rotaxane_md.pdb
.venv/bin/python run_md.py                    # defaults: 0.5 fs, 100 fs, Langevin 300 K
.venv/bin/python run_md.py --time 1000 --dt 1.0 --thermostat nve
.venv/bin/python run_md.py --input rotaxane_displaced_relaxed.xyz
```

`optimize_uma.py` options: `--input`, `--out-xyz`, `--out-pdb`, `--fmax`,
`--steps`, `--smiles` (charge/spin source). `build_rotaxane.py` options:
`--smiles`, `--out`. `run_md.py` options: `--input`, `--out-xyz`, `--out-pdb`,
`--dt` (fs), `--time` (fs), `--thermostat` (langevin|nve), `--temperature` (K),
`--friction` (1/fs), `--stride`, `--flush` (rewrite PDB+XYZ every N steps so a
killed run keeps its trajectory), `--seed`. `displace_wheel.py` options:
`--side` (left|right|farther), `--margin`, `--input`, `--out`.

## Outputs

All geometry outputs are **PyMOL-friendly**: plain standard XYZ (element +
x y z) and multi-state PDB (one state per step; `load` then `mplay` in PyMOL).
ASE's extended XYZ (forces + long `Lattice/Properties` comment) is deliberately
avoided because PyMOL misreads it.

| file | from | content |
|---|---|---|
| `rotaxane.xyz` | build_rotaxane.py | assembled, sterics-optimised |
| `rotaxane_uma_clean.xyz` | optimize_uma.py | UMA-relaxed final frame |
| `rotaxane_uma.pdb` | optimize_uma.py | relaxation trajectory |
| `rotaxane_displaced.xyz` | displace_wheel.py | wheel slid to a steric extreme |
| `rotaxane_displaced_relaxed.xyz` | optimize_uma.py (on displaced) | UMA-relaxed displaced frame (strain-free MD start) |
| `rotaxane_displaced_relax.pdb` | optimize_uma.py (on displaced) | displaced-structure relaxation trajectory |
| `rotaxane_md_clean.xyz` | run_md.py | MD final frame (center-station start) |
| `rotaxane_md.pdb` | run_md.py | MD trajectory (center-station start) |
| `rotaxane_md_displaced_clean.xyz` | run_md.py (on displaced-relaxed) | MD final frame (displaced start) |
| `rotaxane_md_displaced.pdb` | run_md.py (on displaced-relaxed) | MD trajectory (displaced start) |

Generated structure files are gitignored (regenerate by running the scripts).

### A second system

The build and optimize scripts take `--smiles` / `--out` (build) and `--smiles`
(optimize, for charge/spin) so a second SMILES file can be processed without
clobbering the first run's outputs:

```sh
.venv/bin/python build_rotaxane.py --smiles rot_smiles2.txt --out rotaxane2.xyz
.venv/bin/python optimize_uma.py --input rotaxane2.xyz \
    --out-xyz rotaxane2_uma_clean.xyz --out-pdb rotaxane2_uma.pdb \
    --smiles rot_smiles2.txt
```

## Results: full test on `rot_smiles.txt`

Rod = 88 atoms, wheel = 24-crown-8 (56 atoms), 144 atoms total. Rod long
axis ~28 A. All energies from UMA (`uma-s-1p1`, `omol`).

**Geometry relaxation (UMA).** Starting from the sterics-optimised assembly
(`build_rotaxane.py`, 0 steric overlaps), UMA relaxation converges to
E = -124662.31 eV, max|F| = 0.046 eV/A (`rotaxane_uma_clean.xyz`).

**Wheel displacement.** `displace_wheel.py` finds the left stopper at 10.35 A
of clash-free travel (right 7.45 A); the wheel is placed left at 10.05 A
(closest rod-wheel contact 1.19 A). Re-relaxing that strained start settles at
E = -124663.18 eV, max|F| = 0.048 eV/A — **0.87 eV lower** than the centred
station, i.e. the left station is a slightly deeper well
(`rotaxane_displaced_relaxed.xyz`).

**Shuttle MD (5 ps, 500 K Langevin NVT, dt = 1 fs, stride = 5).** Run from
the displaced-relaxed left station:
- Temperature: mean 508 K, range 320-645 K (Langevin bath at 500 K).
- Total energy: mean -124645.3 eV, range ~10.7 eV (NVT, so E_tot fluctuates
  with bath heat exchange rather than being conserved).
- Wheel position along the rod: -9.69 .. -8.43 A about the -9.43 A start, i.e.
  a **~1.0 A rightward excursion** (toward the rod centre) but only ~0.26 A
  leftward — **asymmetric**, because the start sits at the left stopper. A
  sustained rightward slide (peak at ~3.2 ps) is followed by a return.
- Whole-structure RMSD vs frame 0 (Kabsch-aligned): 0.00-2.92 A.
- Outputs: `rotaxane_md_shuttle_clean.xyz` (final frame) and
  `rotaxane_md_shuttle.pdb` (1000-state trajectory; `mplay` in PyMOL).

**Wheel displacement vs temperature.** The instantaneous wheel position is
essentially uncorrelated with the instantaneous temperature
(corr(T, |displacement|) = +0.06). The hottest 10 % of frames (T ~ 570 K) show
only a slightly larger mean excursion (0.44 A) than the coldest 10 %
(T ~ 426 K, 0.35 A). The shuttle is an oscillator whose instantaneous position
is set by its dynamics and the asymmetric steric landscape, not by
moment-to-moment thermal energy.

Plots of temperature, (normalised) energy, wheel displacement, and RMSD are in
`md_temperature.png`, `md_energy.png`, `md_wheel.png`, `md_rmsd.png`, and the
combined `md_overview.png` (regenerate with `plot_md.py`).

## Plots

![MD temperature](md_temperature.png)

![MD energies (normalised: first value = 0)](md_energy.png)

![Wheel shuttle along the rod](md_wheel.png)

![Whole-structure RMSD vs frame 0](md_rmsd.png)

![Overview: 2x2 panel](md_overview.png)

## Credits

- RDKit for 3D embedding and SMILES handling.
- `fairchem-core` / Meta's UMA (`uma-s-1p1`) for energies and forces.
- ASE for optimisation and dynamics.