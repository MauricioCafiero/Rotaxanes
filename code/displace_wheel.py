#!/usr/bin/env python
"""Slide the wheel along the rod: map the shuttle landscape, and place an
MD-start geometry at a side well.

Reads the UMA-relaxed rotaxane (default <stem>_relaxed.xyz), finds the rod's
long axis by PCA, and slides the wheel along +/- that axis. Two things happen:

1. **Stability scan** (the main output): a *chain-seeded* relaxed scan -- two
   monotonic outward sweeps from the central relaxed minimum, each station
   seeded from the previous relaxed geometry so the wheel threads through a
   stopper incrementally (a fresh rigid start past a stopper sits in deep
   overlap and does not converge on CPU). Each sweep walks until it has mapped
   every well on its side and hits the rod-tip wall (energy or contact cutoff,
   rod-length-aware), so multiple wells on a long rod are captured. Produces
   `<stem>_scan.png` (energy vs displacement, kcal/mol, with global-min and
   well markers) and `<stem>_scan.csv`. UMA is a smooth ML potential, so this
   gives a finite shuttle landscape with real minima and barriers as the wheel
   passes over the rod's phenyl/CF3 features -- unlike a classical force field,
   whose 1/r^12 vdW term diverges at the transient close contacts and swamps
   the subtle wells. Needs HF_TOKEN. Use `--no-scan` to skip. The landscape is
   on the wheel-rigid / rod-endpoint-anchored constrained surface, so it
   includes some rod-conformer relaxation; well depths/barriers are on that
   surface, not the free PES.

2. **Well placement** (the MD starting geometry): the scan's wells are detected
   (local minima separated from the global min by a > BARRIER_MIN kcal/mol
   barrier, after smoothing out rod-conformer bumps), and the wheel is placed
   at the chosen well (--side left|right|farther|deeper, default farther =
   largest |d|). The well geometry is already scan-relaxed, so it is written
   to `<stem>_displaced.xyz` ready for `run_md.py` (which does not re-relax).
   If no wells are found, or with `--place-rigid`, the legacy rigid
   stopper-wall placement is used instead (and must be relaxed with
   `optimize_uma.py` before run_md).

The rod is bumpy (phenyl/CF3 groups), so as the wheel slides its closest
contact dips and recovers as features pass through the ring. A real stopper is
therefore a *sustained* overlap: the minimum rod-wheel distance stays below a
floor over a width of several angstroms. We detect the first such wall
(rod-length-aware) for the advisory plot annotation and the rigid fallback.
"""

import argparse
import csv
import math

import numpy as np
from ase.io import read
from rdkit import Chem

from build_rotaxane import read_smiles
from rotaxane_paths import (
    resolve_stem, out_path, default_smiles, OUTPUT_DIR,
    ENGINES, DEFAULT_ENGINE, engine_tag,
)

DEFAULT_IN = out_path("rot_smiles", "relaxed", "xyz")

FLOOR = 1.0        # A; closer than this between a rod-wheel pair = overlap
WALL_WIDTH = 3.0   # A; a stopper is an overlap sustained over this width
SCAN_STEP = 0.05   # A scan increment for stopper detection
MARGIN = 0.3       # A safety back-off from the wall (rigid placement only)
ABS_MAX_SLIDE = 100.0  # A; ultimate safety cap on the stopper search. The real
                      # bound is rod-extent-based (max_slide), so this only
                      # guards against pathological input.

SCAN_GRID = 0.25   # A spacing of the stability-vs-position energy scan (a
                   # relaxed UMA minimisation per grid point). 0.25 A resolves
                   # the ~0.6-A-wide shuttle wells; pass --scan-grid 0.1 for a
                   # publication landscape.
SCAN_PAD = 4.0     # A: extend the --no-scan-chain (rigid) scan past each
                   # stopper. The chain scan ignores this -- it walks until the
                   # tip-wall cutoff (WALK_EMAX / WALK_CONTACT) instead.
SCAN_FMAX = 0.05   # eV/A: force tolerance for the per-point relaxation. Tight
                   # enough to resolve wells/barriers (the old 0.5 default gave
                   # wrong, too-high barriers and missed the deeper minima).
SCAN_STEPS = 200   # max relax steps per grid point (the threading events that
                   # form a side well can take ~195 steps to converge).
SCAN_EMAX = None   # kcal/mol: clip the plot (not the CSV) at min + this; None
                   # = no clip. Set (e.g. --scan-emax 50) only if a stray point
                   # blows up the y-axis.

# Chain-scan walk cutoffs: each outward sweep stops once it has mapped every
# well on its side and is ramming the rod tip wall (rather than grinding into
# the inaccessible, plot-wrecking high-energy tip). Energy-based stops a sweep
# past the last well + its barrier (handles a side with no tip clash before the
# rod end, where the wheel would otherwise dethread); contact-based stops at a
# genuine deep clash. Both are tunable for systems with higher barriers.
WALK_EMAX = 40.0   # kcal/mol above the scan min -- stop a sweep past this
WALK_CONTACT = 1.2  # A -- stop a sweep when min_contact drops below this
BARRIER_MIN = 3.0  # kcal/mol -- a local minimum counts as a well only if it is
                   # separated from the global min by a barrier taller than this
                   # (filters rod-conformer noise bumps that aren't real wells)
SMOOTH_PTS = 5    # moving-average window in POINTS (not A) for well detection.
                   # A point-count window keeps the smoothing width proportional to
                   # the grid: ~1.25 A at the 0.25 default (unchanged behaviour),
                   # shrinking to 0.5 A at 0.1 -- narrow enough to resolve the
                   # ~0.4-A stopper wells that an A-fixed 1.25 A window smears into
                   # the monotonic flank. (An A-fixed window grows in point count as
                   # the grid refines, so finer grids detect stopper wells WORSE,
                   # not better.) Depth is still taken from the raw snapped floor,
                   # so narrowing the window only affects detection, not depth.

# Symmetric (mirror-seeded) chain scan. Both rods are end-to-end chemically
# symmetric (verified via RDKit symmetry-equivalent atom ranks), so the two-
# sweep scan's lopsided landscape (e.g. rot2: barriers 18.8 vs 12.0 kcal/mol on
# mirror sides) is a hysteresis artifact -- each sweep relaxes the rod into a
# different, non-mirror conformer. The symmetric scan builds the -u side as the
# mirror of the +u side so the two agree by construction. A rod conformer pre-
# search avoids the fixed-seed single-embed bias in build_rotaxane (seed
# 0xC0FFEE) that could otherwise seed the d=0 minimum in a bad local-min conformer.
ROD_CONFORMERS = 10   # number of RDKit rod conformers to generate + MMFF-rank for
                      # the symmetric d=0 seed (0 = use the input rod as-is).
SYM_TOL = 0.5         # kcal/mol -- |E(+d) - E(-d)| above this re-relaxes the
                      # higher side seeded from the mirror of the lower one.
SYM_ITERS = 3         # max passes of the symmetry agreement loop (rarely fires --
                      # the mirror seed is already a stationary point of the -d
                      # constrained problem, so it relaxes in ~0 steps).

EV_TO_KCAL_MOL = 23.0605  # 1 eV = 23.0605 kcal/mol (96.485 kJ/mol). UMA returns
                          # eV; scan outputs (CSV + plot + summary) report
                          # relative energies in kcal/mol for chemists.


def fragment_counts(smiles_path):
    """rod / wheel atom counts (with H) from a rod:/wheel: file, matching build."""
    rod_smi, wheel_smi = read_smiles(smiles_path)
    rod_n = Chem.AddHs(Chem.MolFromSmiles(rod_smi)).GetNumAtoms()
    wheel_n = Chem.AddHs(Chem.MolFromSmiles(wheel_smi)).GetNumAtoms()
    return rod_n, wheel_n


def rod_axis(rod_pos):
    """Unit vector along the rod's long axis (largest PCA eigenvalue)."""
    centered = rod_pos - rod_pos.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, np.argmax(eigvals)]


def min_distance(rod_pos, wheel_pos):
    """Closest rod-wheel atom-atom distance (A)."""
    diff = rod_pos[:, None, :] - wheel_pos[None, :, :]
    return float(np.sqrt((diff ** 2).sum(axis=-1)).min())


def mirror_through_rod_center(pos, u, rod_center):
    """Reflect every atom through the plane perpendicular to `u` at `rod_center`.

    Symmetry-equivalent rod atoms share the same element (verified via RDKit
    symmetry-equivalent atom ranks), so reflecting the rod WITHOUT permuting atom
    labels still puts the correct element at every position -- the UMA energy is
    unchanged. The rod/wheel split (first rod_n atoms) is preserved, and the
    FixAtoms tip-anchors are recomputed downstream by argmin/argmax of the
    projection, so they stay correct. The wheel (crown ether) is achiral and
    symmetric, so reflecting it is a valid rigid motion. Thus the mirror of a +d
    relaxed geometry is a stationary point of the -d constrained problem.
    """
    return pos - 2.0 * np.outer((pos - rod_center) @ u, u)


def best_rod_conformer(rod_smi, n_conformers):
    """Lowest-MMFF-energy RDKit conformer of the rod, or None to use the input rod.

    build_rotaxane embeds the rod with a fixed seed (0xC0FFEE), so a single local-
    min conformer can bias the scan. Generate `n_conformers` ETKDGv3 conformers
    (varied seed per index), MMFF-optimize each (UFF fallback, mirroring
    build_rotaxane.mol_to_xyz_block), and return the lowest-energy one's positions
    + elements. The rod backbones are rigid aromatic phenylenes, so the MMFF
    minimum stays extended (the conformer spread is mainly stopper phenyl
    rotations). n_conformers == 0 returns None (caller uses the input rod as-is).
    """
    if n_conformers <= 0:
        return None
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(rod_smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers,
                                          params=params)
    if not conf_ids:
        raise RuntimeError("RDKit rod embedding failed for conformer pre-search")
    best_e, best_pos, best_elems = None, None, None
    for cid in conf_ids:
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid)
            ff = AllChem.MMFFGetMoleculeForceField(mol, confId=cid)
            e = float(ff.CalcEnergy())
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
            e = float(AllChem.UFFGetMoleculeForceField(mol, confId=cid).CalcEnergy())
        if best_e is None or e < best_e:
            best_e = e
            best_pos = np.array(mol.GetConformer(cid).GetPositions(), dtype=float)
            best_elems = [a.GetSymbol() for a in mol.GetAtoms()]
    print(f"rod conformer pre-search: {len(conf_ids)} MMFF conformers, "
          f"lowest energy {best_e:.3f} kcal/mol selected", flush=True)
    return best_elems, best_pos


def sustained_wall(rod_pos, wheel_pos, direction, d0):
    """True if the overlap below FLOOR is sustained over WALL_WIDTH starting d0."""
    for dd in np.arange(d0, d0 + WALL_WIDTH + SCAN_STEP, SCAN_STEP):
        w = wheel_pos + direction * dd
        if min_distance(rod_pos, w) >= FLOOR:
            return False
    return True


def max_slide(rod_pos, wheel_pos, direction, limit=None):
    """Largest shift along `direction` before the first sustained stopper wall.

    A stopper cannot sit past the rod tip, so the search is bounded by the rod's
    extent along `direction` (rod-length-aware) rather than a fixed cap: `limit`
    defaults to how far the wheel centroid can travel before passing the
    farthest rod atom along `direction` plus 1 A. This makes the stopper search
    correct for long rods (the old fixed 25 A cap returned a fake stopper when
    the real wall lay beyond it). Used only for the advisory plot annotation and
    the rigid placement fallback; the chain scan threads past stoppers.
    """
    if limit is None:
        wheel_c = float(wheel_pos.mean(axis=0) @ direction)
        rod_far = float((rod_pos @ direction).max())
        limit = min(ABS_MAX_SLIDE, rod_far - wheel_c + 1.0)
    d = 0.0
    while d <= limit:
        if sustained_wall(rod_pos, wheel_pos, direction, d):
            return max(0.0, d - SCAN_STEP)
        d += SCAN_STEP
    return float(limit)


def write_plain_xyz(path, symbols, pos, comment=""):
    with open(path, "w") as fh:
        fh.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, pos):
            fh.write(f"{s:<2} {x: .6f} {y: .6f} {z: .6f}\n")


# --------------------------------------------------------------------------- #
# Stability-vs-position scan (relaxed UMA per wheel position)
# --------------------------------------------------------------------------- #
def run_scan(symbols, pos0, rod_n, u, left, right, smiles_path,
             grid=SCAN_GRID, pad=SCAN_PAD, fmax=SCAN_FMAX, steps=SCAN_STEPS,
             engine=DEFAULT_ENGINE, method="GFN2-xTB"):
    """Relaxed scan of the wheel along the rod (`engine` is the force source).

    At each grid point the wheel is placed rigidly at `wheel0 + u*d` (station d
    along the rod), then the engine relaxes the structure with:
      - the wheel held RIGID and fixed at station d (FixAtoms), so it stays
        mechanically threaded on the rod and cannot pop off sideways, and
      - the rod's two endpoint atoms anchored (FixAtoms), so the rod can flex
        internally to relieve close contacts but cannot translate/escape.
    So the only degrees of freedom are the rod's internal flex (phenyl/CF3
    groups rotating away from the wheel) -- exactly the clash-relief we want,
    while the scan coordinate (wheel station along the rod) stays fixed. A loose
    convergence (fmax, steps) is used -- just enough to push apart bad sterics,
    not a full minimisation -- so each point is quick even on CPU.

    Returns (ds, energies, min_contacts, pos_by_d, converged). `ds` are signed
    displacements along `u` (positive = +u direction); `energies` are the relaxed
    potential energies (eV); `min_contacts` are the closest rod-wheel
    distances at the *relaxed* geometry; `pos_by_d` is None (the rigid path keeps
    no per-station geometry); `converged` is a bool array (False = hit the step
    cap, an unreliable energy to exclude from well detection).
    """
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS
    from optimize_uma import (
        MODEL, TASK, VACUUM, get_hf_token, make_calculator, read_charge_spin,
    )

    if engine == "uma":
        get_hf_token()
    charge, spin = read_charge_spin(smiles_path)

    atoms = Atoms(symbols=symbols, positions=np.asarray(pos0, dtype=float))
    atoms.set_pbc(False)
    atoms.center(vacuum=VACUUM)  # large non-periodic box; translation-invariant
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    calc, device = make_calculator(engine, atoms, charge, spin, method=method)
    atoms.calc = calc

    if engine == "uma":
        print(f"scan: UMA relaxed  model={MODEL} task={TASK} device={device} "
              f"charge={charge} spin={spin}  fmax={fmax} eV/A  steps={steps}")
    else:
        print(f"scan: tblite relaxed  method={method} charge={charge} spin={spin} "
              f"fmax={fmax} eV/A  steps={steps}")

    # Work in the centered frame. Translation doesn't change `u` or the rod/wheel
    # relative geometry, so the displacement grid is unchanged.
    centered = atoms.get_positions()
    rod_pos = centered[:rod_n]
    wheel_pos0 = centered[rod_n:]
    wheel_idx = list(range(rod_n, len(symbols)))
    # Rod endpoint anchors (global indices; rod is atoms 0..rod_n-1): the atoms
    # at the two extremes along the rod axis. Fixing these lets the rod flex
    # internally but prevents it sliding/escaping under the wheel.
    proj = rod_pos @ u
    anchors = [int(np.argmin(proj)), int(np.argmax(proj))]
    fixed_idx = wheel_idx + anchors

    lo = -left - pad
    hi = right + pad
    ds = np.arange(lo, hi + grid / 2, grid)
    energies = np.empty_like(ds)
    contacts = np.empty_like(ds)
    converged = np.ones(len(ds), dtype=bool)

    for k, d in enumerate(ds):
        pos = centered.copy()
        pos[rod_n:] = wheel_pos0 + u * d
        # Clear the constraint before repositioning: FixAtoms would otherwise
        # pin the wheel to the *previous* point's relaxed position and ignore
        # the new station-d placement (set_positions respects an attached
        # FixAtoms). Place the wheel at station d, then re-attach.
        atoms.set_constraint()
        atoms.set_positions(pos)
        atoms.set_constraint([FixAtoms(fixed_idx)])
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=fmax, steps=steps)
        energies[k] = float(atoms.get_potential_energy())
        relaxed = atoms.get_positions()
        contacts[k] = min_distance(relaxed[:rod_n], relaxed[rod_n:])
        converged[k] = bool(opt.converged())
        # wheel centroid along u relative to rod centroid (should track d;
        # small offset = the rod centroid shifting as the rod flexes)
        wc_u = float((relaxed[rod_n:].mean(axis=0) - relaxed[:rod_n].mean(axis=0)) @ u)
        flag = "converged" if converged[k] else "NOT-converged"
        print(f"  scan {k + 1}/{len(ds)}: d={d:+.2f} A  "
              f"E={energies[k]:.4f} eV  min_contact={contacts[k]:.2f} A  "
              f"wheel_u={wc_u:+.2f} A  (relax steps={opt.get_number_of_steps()})  "
              f"{flag}",
              flush=True)
    return ds, energies, contacts, None, converged  # no per-station positions kept


# --------------------------------------------------------------------------- #
# Relaxed-chain scan: two monotonic outward sweeps from the central minimum
# --------------------------------------------------------------------------- #
def run_scan_chain(symbols, pos0, rod_n, u, smiles_path,
                   grid=SCAN_GRID, fmax=SCAN_FMAX, steps=SCAN_STEPS,
                   walk_emax=WALK_EMAX, walk_contact=WALK_CONTACT,
                   engine=DEFAULT_ENGINE, method="GFN2-xTB"):
    """Relaxed scan (`engine` force source) as two outward sweeps from the
    central minimum, each walking until it has mapped every well on its side and
    hits the rod tip wall.

    Both sweeps start from the input relaxed geometry (the d=0 seed) and step the
    wheel outward along the rod axis by `grid` at a time. Every station is seeded
    from the previous point's relaxed geometry -- the wheel is nudged rigidly
    one grid step along u, the rod is carried forward -- then the engine relaxes
    with the wheel held rigid (FixAtoms) and the rod's two endpoint atoms
    anchored (so the rod can flex internally to relieve sterics but cannot
    translate or escape). Seeding from the prior relaxed structure lets the
    wheel thread through a stopper incrementally so each point converges in a
    few steps; a fresh rigid start past a stopper sits in deep overlap and does
    not converge on CPU (run_scan).

    Each sweep stops once it is past the last well and into the tip wall:
      - the relaxed energy rises more than `walk_emax` (kcal/mol) above the
        running global minimum, or
      - the closest rod-wheel contact drops below `walk_contact` (a genuine
        deep clash with the rod tip), or
      - the wheel centroid passes the farthest rod atom along that direction
        (rod-length-aware; beyond it the wheel has dethreaded).
    This maps every station on a side -- including multiple wells on a long rod
    -- and stops before the inaccessible, plot-wrecking high-energy tip.

    Returns (ds, energies, contacts, pos_by_d, converged) sorted by signed
    displacement d along u (d=0 is the input relaxed geometry). energies are
    relaxed potential energies (eV); contacts are the closest rod-wheel
    distances at the relaxed geometry; pos_by_d maps each d to its relaxed
    positions (np.array, same atom order as the input) so a well's geometry can
    be written out as an MD start without re-relaxing; converged is a bool array
    (False = hit the step cap during a threading event, an unreliable energy).
    """
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS
    from optimize_uma import (
        MODEL, TASK, VACUUM, get_hf_token, make_calculator, read_charge_spin,
    )

    if engine == "uma":
        get_hf_token()
    charge, spin = read_charge_spin(smiles_path)

    atoms = Atoms(symbols=symbols, positions=np.asarray(pos0, dtype=float))
    atoms.set_pbc(False)
    atoms.center(vacuum=VACUUM)  # large non-periodic box; translation-invariant
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    calc, device = make_calculator(engine, atoms, charge, spin, method=method)
    atoms.calc = calc

    if engine == "uma":
        print(f"scan(chain): UMA relaxed  model={MODEL} task={TASK} device={device} "
              f"charge={charge} spin={spin}  fmax={fmax} eV/A  steps={steps}  "
              f"grid={grid} A  walk_emax={walk_emax} kcal/mol  "
              f"walk_contact={walk_contact} A  (two sweeps from d=0)", flush=True)
    else:
        print(f"scan(chain): tblite relaxed  method={method} charge={charge} "
              f"spin={spin}  fmax={fmax} eV/A  steps={steps}  grid={grid} A  "
              f"walk_emax={walk_emax} kcal/mol  walk_contact={walk_contact} A  "
              f"(two sweeps from d=0)", flush=True)

    centered = atoms.get_positions()
    rod_pos = centered[:rod_n]
    wheel0 = centered[rod_n:]
    wheel_idx = list(range(rod_n, len(symbols)))
    proj = rod_pos @ u
    anchors = [int(np.argmin(proj)), int(np.argmax(proj))]
    fixed_idx = wheel_idx + anchors
    # Rod-extent hard cap per direction (rod-length-aware): past the farthest rod
    # atom the wheel has dethreaded, so stop.
    wheel_c = float(wheel0.mean(axis=0) @ u)
    cap_right = min(ABS_MAX_SLIDE, float(proj.max()) - wheel_c + 1.0)
    cap_left = min(ABS_MAX_SLIDE, wheel_c - float(proj.min()) + 1.0)

    def relax_at(positions, d):
        atoms.set_constraint()           # clear before repositioning (see run_scan)
        atoms.set_positions(positions)
        atoms.set_constraint([FixAtoms(fixed_idx)])
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=fmax, steps=steps)
        relaxed = atoms.get_positions()
        e = float(atoms.get_potential_energy())
        c = min_distance(relaxed[:rod_n], relaxed[rod_n:])
        wc_u = float((relaxed[rod_n:].mean(axis=0) - relaxed[:rod_n].mean(axis=0)) @ u)
        flag = "converged" if opt.converged() else "NOT-converged"
        print(f"  scan d={d:+.2f} A  E={e:.4f} eV  min_contact={c:.2f} A  "
              f"wheel_u={wc_u:+.2f} A  relax_steps={opt.get_number_of_steps()}  "
              f"{flag}", flush=True)
        return relaxed, e, c, bool(opt.converged())

    # d=0 seed: the input relaxed geometry, relaxed once under the wheel-rigid /
    # rod-endpoint-anchored constraints (converges in 0 steps -- already minimum).
    seed, e0, c0, conv0 = relax_at(centered, 0.0)
    results = {0.0: (e0, c0)}
    pos_by_d = {0.0: seed.copy()}
    conv_by_d = {0.0: conv0}
    emin = e0  # running global min across both sweeps; the energy cutoff is
               # measured above this so deep-min systems still stop at the tip

    for sign, cap in ((+1.0, cap_right), (-1.0, cap_left)):
        state = seed.copy()
        k = 1
        while k * grid <= cap:
            d = sign * k * grid
            pos = state.copy()
            pos[rod_n:] = state[rod_n:] + (sign * u) * grid  # nudge wheel one step
            relaxed, e, c, conv = relax_at(pos, d)
            results[d] = (e, c)
            pos_by_d[d] = relaxed.copy()
            conv_by_d[d] = conv
            state = relaxed
            emin = min(emin, e)
            # Stop past the last well, into the tip wall.
            if (e - emin) * EV_TO_KCAL_MOL > walk_emax or c < walk_contact:
                print(f"  sweep {'+' if sign > 0 else '-'}u stops at d={d:+.2f} A "
                      f"(E_rel={(e - emin) * EV_TO_KCAL_MOL:.1f} kcal/mol, "
                      f"min_contact={c:.2f} A)", flush=True)
                break
            k += 1

    ds = np.array(sorted(results))
    energies = np.array([results[d][0] for d in ds])
    contacts = np.array([results[d][1] for d in ds])
    converged = np.array([conv_by_d[d] for d in ds])
    n_bad = int((~converged).sum())
    if n_bad:
        print(f"scan(chain): {n_bad} station(s) NOT converged -- excluded from "
              f"well detection (use --production for more relax steps)", flush=True)
    return ds, energies, contacts, pos_by_d, converged


def rotation_align(a, b):
    """Rotation matrix mapping unit vector `a` onto unit vector `b` (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(a @ b)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180-degree flip about any axis perpendicular to a
        n = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        a = a - (a @ n) * n
        a /= np.linalg.norm(a)
        c = float(a @ b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kx + (kx @ kx) * ((1 - c) / (s * s))


# --------------------------------------------------------------------------- #
# Symmetric (mirror-seeded) chain scan: build the -u side as the mirror of +u
# --------------------------------------------------------------------------- #
def run_scan_chain_sym(symbols, pos0, rod_n, u, smiles_path,
                       grid=SCAN_GRID, fmax=SCAN_FMAX, steps=SCAN_STEPS,
                       walk_emax=WALK_EMAX, walk_contact=WALK_CONTACT,
                       rod_conformers=ROD_CONFORMERS, sym_tol=SYM_TOL,
                       sym_iters=SYM_ITERS,
                       engine=DEFAULT_ENGINE, method="GFN2-xTB"):
    """Relaxed scan (`engine` force source) whose landscape is mirror-symmetric
    about d=0 by construction -- the fix for the two-sweep scan's hysteresis
    asymmetry.

    Both rods are end-to-end chemically symmetric (verified via RDKit symmetry-
    equivalent atom ranks), so the two-sweep scan's lopsided landscape (e.g. rot2:
    barriers 18.8 vs 12.0 kcal/mol on mirror sides) is an artifact of each sweep
    relaxing the rod into a different, non-mirror conformer. Here only the +u side
    is swept (chain-seeded, threading the stopper incrementally as in
    run_scan_chain); the -u side is built by *mirroring* each +d relaxed geometry
    through the rod-center plane (mirror_through_rod_center) and re-relaxing. The
    mirror of a +d geometry is a stationary point of the -d constrained problem
    (symmetry-equivalent atoms share an element, so reflection preserves the
    energy; the wheel is achiral/symmetric; the FixAtoms tip-anchors are
    recomputed by argmin/argmax), so it relaxes in ~0 steps and E(-d) = E(+d) to
    optimizer tolerance -- no atom permutation is needed.

    A rod conformer pre-search (best_rod_conformer) replaces the fixed-seed single
    embed from build_rotaxane with the lowest-MMFF-energy of N RDKit conformers,
    aligned to the input frame (long axis -> u, centered at the rod center), then
    engine-optimized (a free rod relaxation with the wheel held rigid at center) so
    the stopper phenyls reach a proper engine minimum. A raw MMFF conformer used
    as-is lands in a local minimum whose stopper orientation blocks threading; the
    free pre-relax makes the conformer-derived rod comparable to the already-
    relaxed input rod.

    An agreement loop (sym_tol, sym_iters) re-relaxes the higher-energy side of any
    |d| pair seeded from the mirror of the lower side, until |E(+d) - E(-d)| <
    sym_tol -- the user's "re-optimize the lower side until they agree". Rarely
    fires (the mirror seed is already a stationary point); it is the safety net
    for noisy/NOT-converged stations.

    d=0 is the rod center (wheel centered on the rod), so a symmetric rod's global
    minimum lands at d=0. Returns the same (ds, energies, contacts, pos_by_d,
    converged) tuple as run_scan_chain for drop-in use by detect_wells/plot/csv.
    """
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS
    from optimize_uma import (
        MODEL, TASK, VACUUM, get_hf_token, make_calculator, read_charge_spin,
    )

    if engine == "uma":
        get_hf_token()
    charge, spin = read_charge_spin(smiles_path)

    atoms = Atoms(symbols=symbols, positions=np.asarray(pos0, dtype=float))
    atoms.set_pbc(False)
    atoms.center(vacuum=VACUUM)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    calc, device = make_calculator(engine, atoms, charge, spin, method=method)
    atoms.calc = calc

    if engine == "uma":
        print(f"scan(symmetric): UMA relaxed  model={MODEL} task={TASK} "
              f"device={device} charge={charge} spin={spin}  fmax={fmax} eV/A  "
              f"steps={steps}  grid={grid} A  walk_emax={walk_emax} kcal/mol  "
              f"walk_contact={walk_contact} A  rod_conformers={rod_conformers}  "
              f"sym_tol={sym_tol} kcal/mol  sym_iters={sym_iters}  "
              f"(+u swept, -u mirrored)", flush=True)
    else:
        print(f"scan(symmetric): tblite relaxed  method={method} charge={charge} "
              f"spin={spin}  fmax={fmax} eV/A  steps={steps}  grid={grid} A  "
              f"walk_emax={walk_emax} kcal/mol  walk_contact={walk_contact} A  "
              f"rod_conformers={rod_conformers}  sym_tol={sym_tol} kcal/mol  "
              f"sym_iters={sym_iters}  (+u swept, -u mirrored)", flush=True)

    centered = atoms.get_positions()
    rod_pos = centered[:rod_n].copy()
    wheel0 = centered[rod_n:]
    wheel_idx = list(range(rod_n, len(symbols)))
    rod_center = rod_pos.mean(axis=0)

    # Optional rod conformer pre-search: lowest-MMFF RDKit conformer, aligned to
    # the input frame (long axis -> u, centered at the rod center), then UMA-
    # optimized (free rod relaxation with the wheel held rigid at center) so the
    # stopper phenyls reach a proper UMA minimum. A raw MMFF conformer used as-is
    # lands in a local UMA minimum whose stopper orientation blocks threading; the
    # free pre-relax makes the conformer-derived rod comparable to the (already
    # UMA-relaxed) input rod. The conformer is in SMILES atom order, matching the
    # input rod, so the rod/wheel split and atom order are unchanged.
    if rod_conformers > 0:
        rod_smi, _ = read_smiles(smiles_path)
        elems, rconf = best_rod_conformer(rod_smi, rod_conformers)
        if elems != list(symbols[:rod_n]):
            raise RuntimeError(
                "rod conformer element order does not match the input geometry")
        v = rod_axis(rconf)
        rconf = (rotation_align(v, u) @ rconf.T).T
        rconf = rconf - rconf.mean(axis=0) + rod_center
        # UMA-optimize the conformer: place it with the wheel centered, hold the
        # wheel rigid, and let the whole rod relax freely (no tip anchors) so the
        # stoppers find a proper UMA minimum. The centered wheel + symmetric rod
        # give zero net force on the rod, so rod_center stays put.
        pre_pos = centered.copy()
        pre_pos[:rod_n] = rconf
        pre_pos[rod_n:] = wheel0 - wheel0.mean(axis=0) + rod_center
        atoms.set_constraint()
        atoms.set_positions(pre_pos)
        atoms.set_constraint([FixAtoms(wheel_idx)])
        pre_opt = LBFGS(atoms, logfile=None)
        pre_opt.run(fmax=fmax, steps=steps)
        rod_pos = atoms.get_positions()[:rod_n].copy()
        rod_center = rod_pos.mean(axis=0)   # symmetry plane (recomputed; ~unchanged)
        _eng_lbl = "UMA" if engine == "uma" else "tblite"
        print(f"  conformer {_eng_lbl} pre-relax: {pre_opt.get_number_of_steps()} "
              f"steps  E={atoms.get_potential_energy():.4f} eV  "
              f"{'converged' if pre_opt.converged() else 'NOT-converged'}", flush=True)

    proj = rod_pos @ u
    anchors = [int(np.argmin(proj)), int(np.argmax(proj))]
    fixed_idx = wheel_idx + anchors
    # The symmetry plane is fixed at the rod center for the whole scan (the rod
    # flexes, but the symmetry element does not move). caps are rod-length-aware;
    # for a symmetric rod centered at rod_center, cap_left == cap_right, so
    # mirroring the +u sweep maps the whole -u side.
    wheel_c = float(rod_center @ u)
    cap_right = min(ABS_MAX_SLIDE, float(proj.max()) - wheel_c + 1.0)

    def relax_at(positions, d):
        atoms.set_constraint()           # clear before repositioning (see run_scan)
        atoms.set_positions(positions)
        atoms.set_constraint([FixAtoms(fixed_idx)])
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=fmax, steps=steps)
        relaxed = atoms.get_positions()
        e = float(atoms.get_potential_energy())
        c = min_distance(relaxed[:rod_n], relaxed[rod_n:])
        wc_u = float((relaxed[rod_n:].mean(axis=0) - relaxed[:rod_n].mean(axis=0)) @ u)
        flag = "converged" if opt.converged() else "NOT-converged"
        print(f"  scan d={d:+.2f} A  E={e:.4f} eV  min_contact={c:.2f} A  "
              f"wheel_u={wc_u:+.2f} A  relax_steps={opt.get_number_of_steps()}  "
              f"{flag}", flush=True)
        return relaxed, e, c, bool(opt.converged())

    # d=0 seed: wheel centered on the rod (the symmetry origin), rod as prepared
    # (input rod, or the best conformer). Relaxed under the wheel-rigid /
    # rod-tip-anchored constraints. d=0 is its own mirror, so the landscape is
    # symmetric about it regardless of this conformer's exact shape; the conformer
    # pre-search just makes it a fresh low-strain near-symmetric seed.
    seed_pos = centered.copy()
    seed_pos[:rod_n] = rod_pos
    seed_pos[rod_n:] = wheel0 - wheel0.mean(axis=0) + rod_center
    seed, e0, c0, conv0 = relax_at(seed_pos, 0.0)
    pos_by_d = {0.0: seed.copy()}
    e_by_d = {0.0: e0}
    c_by_d = {0.0: c0}
    conv_by_d = {0.0: conv0}
    emin = e0  # running global min; the +u walk cutoff is measured above this

    # +u chain sweep (only one side is swept -- the other is mirrored).
    state = seed.copy()
    k = 1
    while k * grid <= cap_right:
        d = k * grid
        pos = state.copy()
        pos[rod_n:] = state[rod_n:] + u * grid       # nudge wheel one +grid step
        relaxed, e, c, conv = relax_at(pos, d)
        pos_by_d[d] = relaxed.copy()
        e_by_d[d] = e
        c_by_d[d] = c
        conv_by_d[d] = conv
        state = relaxed
        emin = min(emin, e)
        if (e - emin) * EV_TO_KCAL_MOL > walk_emax or c < walk_contact:
            print(f"  sweep +u stops at d={d:+.2f} A  "
                  f"(E_rel={(e - emin) * EV_TO_KCAL_MOL:.1f} kcal/mol, "
                  f"min_contact={c:.2f} A)", flush=True)
            break
        k += 1

    # -u side: mirror each +d relaxed geometry through the rod-center plane and
    # relax under the -d constraints. The mirror is a stationary point of the -d
    # problem, so this converges in ~0 steps and gives E(-d) = E(+d).
    pos_ds = sorted(d for d in pos_by_d if d > 0.0)
    for d in pos_ds:
        m = mirror_through_rod_center(pos_by_d[d], u, rod_center)
        relaxed, e, c, conv = relax_at(m, -d)
        pos_by_d[-d] = relaxed.copy()
        e_by_d[-d] = e
        c_by_d[-d] = c
        conv_by_d[-d] = conv

    # Agreement loop: for any |d| pair disagreeing by > sym_tol, re-relax the
    # higher side seeded from the mirror of the lower side. (Rarely fires -- the
    # mirror seed is already a stationary point; this is the safety net for noisy
    # / NOT-converged stations.)
    for _ in range(sym_iters):
        worst_d, worst_gap = None, 0.0
        for d in pos_ds:
            gap = abs(e_by_d[d] - e_by_d[-d]) * EV_TO_KCAL_MOL
            if gap > worst_gap:
                worst_d, worst_gap = d, gap
        if worst_d is None or worst_gap <= sym_tol:
            break
        d = worst_d
        if e_by_d[d] <= e_by_d[-d]:
            # + side is lower (or tied): seed -d from mirror(+d), re-relax the - side
            src = mirror_through_rod_center(pos_by_d[d], u, rod_center)
            relaxed, e, c, conv = relax_at(src, -d)
            pos_by_d[-d], e_by_d[-d], c_by_d[-d], conv_by_d[-d] = \
                relaxed.copy(), e, c, conv
        else:
            # - side is lower: seed +d from mirror(-d), re-relax the + side
            src = mirror_through_rod_center(pos_by_d[-d], u, rod_center)
            relaxed, e, c, conv = relax_at(src, d)
            pos_by_d[d], e_by_d[d], c_by_d[d], conv_by_d[d] = \
                relaxed.copy(), e, c, conv
        new_gap = abs(e_by_d[d] - e_by_d[-d]) * EV_TO_KCAL_MOL
        print(f"  sym agreement: |d|={d:.2f} A gap {worst_gap:.2f} -> "
              f"{new_gap:.2f} kcal/mol (re-relaxed the higher side)", flush=True)

    ds = np.array(sorted(e_by_d))
    energies = np.array([e_by_d[d] for d in ds])
    contacts = np.array([c_by_d[d] for d in ds])
    converged = np.array([conv_by_d[d] for d in ds])
    # Report the residual symmetry agreement so the user can see it held.
    gaps = [abs(e_by_d[d] - e_by_d[-d]) * EV_TO_KCAL_MOL for d in pos_ds]
    max_gap = max(gaps) if gaps else 0.0
    print(f"scan(symmetric): max |E(+d) - E(-d)| = {max_gap:.3f} kcal/mol over "
          f"{len(pos_ds)} mirror pairs", flush=True)
    n_bad = int((~converged).sum())
    if n_bad:
        print(f"scan(symmetric): {n_bad} station(s) NOT converged -- excluded "
              f"from well detection (use --production for more relax steps)",
              flush=True)
    return ds, energies, contacts, pos_by_d, converged


def detect_wells(ds, energies, grid, pos_by_d, converged=None,
                 smooth_pts=SMOOTH_PTS):
    """Identify shuttle wells (metastable side minima) on the scan curve.

    The chain landscape carries rod-conformer relaxation noise (~0.5-2
    kcal/mol bumps), so we smooth before picking minima: a moving average over a
    POINT-count window (smooth_pts, clamped >= 3 and odd). The window is in
    points, not A, so it scales with the grid: ~1.25 A at 0.25 (the default,
    unchanged) shrinking to 0.5 A at 0.1 -- narrow enough to resolve the ~0.4-A
    stopper wells. (An A-fixed window would grow in point count as the grid
    refines and smear narrow wells worse at finer grids.) A local minimum on the
    smoothed curve (lower than its +/-2 neighbours) is a candidate. Candidates
    within 1.0 A of each other are merged to the deepest. Smoothing only locates
    the basin (robust to noise); each basin is then *snapped* to its deepest
    CONVERGED raw point, so the window width never biases the reported depth.

    A candidate counts as a *well* only if it is separated from the global
    minimum by a real barrier: the maximum energy on the path between the
    candidate and the global min must exceed the candidate's energy by more
    than BARRIER_MIN (kcal/mol). This filters rod-conformer noise bumps in the
    central region, which sit in a shallow basin with no barrier. The global
    minimum itself is reported separately, never as a well.

    `converged` (bool array, default all-True) marks which stations reached the
    fmax tolerance. NOT-converged stations (a wheel that hit the step cap mid-
    threading) report an unreliable energy -- wherever LBFGS last landed, not a
    minimum -- that can be spuriously low and poison the smoothing. Such points
    are linearly interpolated from their converged neighbours before smoothing,
    are barred from being a candidate/global-min, and the global minimum is
    taken over converged stations only. (For a reporting-quality landscape, run
    with --production so threading events actually converge.)

    Returns a list of wells (nearest the global min first), each a dict:
      {d, e, rel_kcal, side ("left"/"right" of the global min d),
       barrier_kcal, positions}. `positions` is the relaxed geometry at that d
      (np.array, same atom order as the scan input) for use as an MD start.
    """
    ds = np.asarray(ds, dtype=float)
    energies = np.asarray(energies, dtype=float)
    n = len(energies)
    if converged is None:
        converged = np.ones(n, dtype=bool)
    else:
        converged = np.asarray(converged, dtype=bool)

    # Global minimum over CONVERGED stations only: a NOT-converged point can
    # sit spuriously low and must not be mistaken for the basin floor.
    good = np.where(converged)[0]
    if good.size:
        g_idx = int(good[np.argmin(energies[good])])
    else:
        g_idx = int(np.argmin(energies))
    e_min = float(energies[g_idx])
    d_min = float(ds[g_idx])

    # Replace NOT-converged energies with a linear interpolation from converged
    # neighbours so a spurious dip/spike can't poison the smoothing or the
    # barrier height. `work` is the curve used for smoothing + barrier search;
    # the raw `energies` is still used for a well's own depth (wells are only at
    # converged stations, so their raw energy is reliable).
    work = energies.copy()
    bad = np.where(~converged)[0]
    if bad.size:
        work[bad] = np.interp(bad, good, energies[good])

    # Smooth over a POINT-count window (odd, >= 3) to suppress rod-conformer
    # bumps. Point-count (not A) so the window shrinks with the grid and can
    # resolve the ~0.4-A stopper wells at fine grids (an A-fixed window would
    # smear them worse as the grid refines). Depth comes from the raw snapped
    # floor below, so this only affects detection.
    w = max(int(smooth_pts), 3)
    if w % 2 == 0:
        w += 1
    half = w // 2
    smooth = np.empty_like(work)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smooth[i] = work[lo:hi].mean()

    # Local minima on the smoothed curve (window +/-2 points), but only at
    # converged stations (a NOT-converged point can't be a well).
    candidates = []
    for i in range(2, n - 2):
        if not converged[i]:
            continue
        if smooth[i] < smooth[i - 1] and smooth[i] < smooth[i + 1] \
           and smooth[i] <= smooth[i - 2] and smooth[i] <= smooth[i + 2]:
            candidates.append(i)
    # Merge candidates within 1.0 A, keeping the deepest at the raw (unsmoothed)
    # energy so well depths aren't biased by smoothing.
    merged = []
    for i in candidates:
        d = ds[i]
        if merged and abs(d - ds[merged[-1]]) <= 1.0:
            if energies[i] < energies[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)

    wells = []
    for i in merged:
        if i == g_idx:
            continue  # global min, not a well
        # Snap to the true raw minimum of the basin: smoothing only locates the
        # basin (robust to rod-conformer noise), but its window (1.25 A) blurs
        # the well with the adjacent barrier top and shifts the apparent minimum
        # off the real floor by up to one grid step. Report the deepest CONVERGED
        # raw point within +/-half of the smoothed candidate as the well.
        lo_s, hi_s = max(0, i - half), min(n, i + half + 1)
        window = np.arange(lo_s, hi_s)
        window = window[converged[window]]
        if window.size == 0:
            continue
        j = int(window[int(np.argmin(energies[window]))])
        if j == g_idx:
            continue  # global min, not a well
        d = float(ds[j])
        e = float(energies[j])
        # Barrier = highest point on the path between this min and the global min
        # (on the interpolated curve, so a NOT-converged spike can't fake one).
        lo, hi = (j, g_idx) if j < g_idx else (g_idx, j)
        barrier = float(work[lo:hi + 1].max())
        barrier_kcal = (barrier - e) * EV_TO_KCAL_MOL
        if barrier_kcal < BARRIER_MIN:
            continue  # not separated by a real barrier -> noise bump
        side = "left" if d < d_min else "right"
        wells.append({
            "d": d,
            "e": e,
            "rel_kcal": float((e - e_min) * EV_TO_KCAL_MOL),
            "side": side,
            "barrier_kcal": barrier_kcal,
            "positions": pos_by_d[d] if pos_by_d is not None else None,
        })
    # Nearest-to-global-min first (smallest |d - d_min|), then by depth.
    wells.sort(key=lambda w: (abs(w["d"] - d_min), w["rel_kcal"]))
    return wells, d_min, e_min


def plot_scan(ds, energies, contacts, left, right, place_d, place_side,
              out_png, emax=SCAN_EMAX, wells=None, d_min=None, e_min=None,
              engine=DEFAULT_ENGINE):
    """Write the energy-vs-position PNG. Energy is plotted relative to its min
    in kcal/mol; if `emax` is set (kcal/mol), the plot (not the CSV) is clipped
    at min + emax so a stray point doesn't flatten the landscape. The global
    minimum, each detected well, the (advisory) rigid stopper walls, and the
    placed station are marked. `e_min` is the converged-only global min (so a
    spurious NOT-converged low point doesn't shift the scale); defaults to
    energies.min() when not supplied. `engine` labels the curve honestly (UMA
    vs tblite/<method>).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if e_min is None:
        e_min = float(np.asarray(energies).min())
    rel = (energies - e_min) * EV_TO_KCAL_MOL  # relative energy, kcal/mol
    disp = np.clip(rel, None, emax) if emax is not None else rel

    eng_label = "UMA" if engine == "uma" else "tblite"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ds, disp, "o-", ms=3, lw=1.0, color="tab:blue",
            label=f"{eng_label} relaxed energy, rel. to min")
    # Rigid stopper walls are advisory only: the chain scan threads past them, so
    # they just mark where a fresh rigid placement would first clash.
    ax.axvline(right, color="tab:red", ls="--", lw=1.0,
              label=f"rigid stopper (first wall) +{right:.2f} A")
    ax.axvline(-left, color="tab:red", ls=":", lw=1.0,
              label=f"rigid stopper (first wall) -{-left:.2f} A")
    if d_min is not None:
        # Index of the converged global min (NOT energies.argmin, which could be
        # a spurious NOT-converged point).
        j = int(np.where(np.isclose(ds, d_min))[0][0]) \
            if any(np.isclose(ds, d_min)) else int(np.argmin(energies))
        ax.plot(d_min, disp[j], "*", ms=12, color="tab:purple",
                label=f"global min (d={d_min:+.2f} A)")
    if wells:
        for k, w in enumerate(wells):
            # plot the well at its own displacement's clipped value
            kk = int(np.where(np.isclose(ds, w["d"]))[0][0]) \
                if any(np.isclose(ds, w["d"])) else None
            yy = disp[kk] if kk is not None else w["rel_kcal"]
            ax.plot(w["d"], yy, "s", ms=8, mfc="none", mec="tab:orange",
                    label=f"well {w['side']} (d={w['d']:+.2f} A, "
                          f"+{w['rel_kcal']:.1f} kcal/mol)")
    if place_d is not None:
        ax.axvline(place_d, color="tab:green", lw=1.2,
                   label=f"placed {place_side} ({place_d:.2f} A)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("wheel displacement along rod axis (A)")
    ax.set_ylabel("energy - E_min  (kcal/mol)")
    ax.set_title("Rotaxane shuttle stability vs wheel position (relaxed)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def write_scan_csv(path, ds, energies, contacts, e_min=None):
    """CSV with displacement, absolute engine energy (eV, for traceability), the
    relative energy in kcal/mol (the chemist-facing column), and the closest
    rod-wheel contact. `e_min` is the reference for the relative column; it
    defaults to energies.min() but should be the converged-only global min when
    NOT-converged stations are present (so a spurious low point doesn't shift
    the scale)."""
    if e_min is None:
        e_min = float(np.asarray(energies).min())
    rel_kcal = (energies - e_min) * EV_TO_KCAL_MOL
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["displacement_A", "energy_eV", "energy_rel_kcal_mol",
                    "min_contact_A"])
        for d, e, rk, c in zip(ds, energies, rel_kcal, contacts):
            w.writerow([f"{d:.4f}", f"{e:.6f}", f"{rk:.4f}", f"{c:.4f}"])


# --------------------------------------------------------------------------- #
# Eyring TST rate estimates from the scan's (potential-energy) barriers
# --------------------------------------------------------------------------- #
RATE_TEMP = 300.0  # K, default for the Eyring rate estimate
KB_HZ = 2.083661912e10   # k_B/h in s^-1 K^-1 -> kBT/h = KB_HZ * T
R_KCAL = 1.98720425864083e-3  # R in kcal/(mol K) -> RT = R_KCAL * T


def rate_estimate(dg_kcal, temperature):
    """Eyring TST rate k (s^-1) and characteristic time tau (s) for a barrier
    `dg_kcal` (kcal/mol) at `temperature` (K): k = (kBT/h) exp(-dg/RT).

    This treats a constrained-surface POTENTIAL-energy barrier as a free-energy
    barrier and uses the Eyring prefactor -- an order-of-magnitude ESTIMATE, not
    a prediction. The scan holds the wheel rigid and anchors the rod endpoints,
    so the path is not the minimum-energy path and carries no entropy; in
    solution the shuttling coordinate is friction-controlled (Kramers), so the
    prefactor is not kBT/h. Use the numbers to rank rates / expose asymmetry,
    not as absolute rates. Returns (k, tau).
    """
    rt = R_KCAL * temperature
    kbt_h = KB_HZ * temperature
    k = kbt_h * math.exp(-dg_kcal / rt)
    tau = 1.0 / k if k > 0 else float("inf")
    return k, tau


def _fmt_time(tau):
    """Human-readable characteristic time across ns..yr."""
    if not math.isfinite(tau):
        return "inf"
    if tau < 1e-6:
        return f"{tau * 1e9:.2g} ns"
    if tau < 1e-3:
        return f"{tau * 1e6:.2g} us"
    if tau < 1:
        return f"{tau * 1e3:.2g} ms"
    if tau < 60:
        return f"{tau:.2g} s"
    if tau < 3600:
        return f"{tau / 60:.2g} min"
    if tau < 86400:
        return f"{tau / 3600:.2g} h"
    if tau < 3.15576e7:
        return f"{tau / 86400:.2g} d"
    return f"{tau / 3.15576e7:.2g} yr"


def _fmt_rate(k):
    """Human-readable rate in s^-1."""
    if k <= 0 or not math.isfinite(k):
        return "0 s^-1"
    exp = int(math.floor(math.log10(k)))
    mant = k / 10 ** exp
    return f"{mant:.2f}e{exp} s^-1"


def print_rate_estimates(wells, temperature=RATE_TEMP):
    """Print Eyring TST rate estimates for each well's escape (well -> global
    min, over the well's barrier) and entry (global min -> well, over the same
    saddle, i.e. rel_kcal + barrier_kcal), with the estimate caveat stamped.
    """
    rt = R_KCAL * temperature
    kbt_h = KB_HZ * temperature
    print(f"rate estimates (Eyring TST, {temperature:.0f} K):  kBT/h={kbt_h:.2e} "
          f"s^-1  RT={rt:.3f} kcal/mol")
    print("  ESTIMATE only -- constrained-surface potential barriers treated as "
          "free-energy barriers with the Eyring prefactor; true rates need a "
          "free-energy profile and (in solution) friction (Kramers).")
    if not wells:
        print("  no wells -> no rate estimates")
        return
    for w in wells:
        escape = w["barrier_kcal"]                       # well -> saddle -> center
        entry = w["rel_kcal"] + w["barrier_kcal"]        # center -> saddle -> well
        k_e, t_e = rate_estimate(escape, temperature)
        k_i, t_i = rate_estimate(entry, temperature)
        print(f"  {w['side']:5s} well (d={w['d']:+.2f} A):  "
              f"escape->center  dG={escape:5.1f} kcal/mol -> k={_fmt_rate(k_e)} "
              f"(tau {_fmt_time(t_e)});  center->well entry  dG={entry:5.1f} -> "
              f"k={_fmt_rate(k_i)} (tau {_fmt_time(t_i)})")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_IN,
                   help="relaxed rotaxane geometry (default: <stem>_relaxed.xyz). "
                        "Outputs are named from this file's stem.")
    p.add_argument("--side", choices=["left", "right", "farther", "deeper"],
                   default="farther",
                   help="which detected well to place the wheel at for the MD "
                        "start: left/right = furthest well on that side; "
                        "farther (default) = the well with the largest |d|; "
                        "deeper = the lowest-energy well. Falls back to the "
                        "rigid stopper-wall placement if no wells are found.")
    p.add_argument("--margin", type=float, default=MARGIN,
                   help="A back-off from the stopper wall (default 0.3) -- used "
                        "only by the rigid fallback placement (--place-rigid or "
                        "no wells found)")
    p.add_argument("--place-rigid", action="store_true",
                   help="skip well placement; use the legacy rigid "
                        "stopper-wall placement (wheel at travel - margin). "
                        "The rigid geometry is strained, so relax it next with "
                        "optimize_uma.py before run_md.py.")
    p.add_argument("--out", default=None,
                   help="displaced-isomer XYZ (default: <stem>_displaced.xyz)")
    p.add_argument("--smiles", default=None,
                   help="rod:/wheel: file for atom counts + charge/spin "
                        "(default: <stem>.txt matching the input)")
    p.add_argument("--engine", default=DEFAULT_ENGINE, choices=ENGINES,
                   help="force source for the scan relaxations: 'uma' (Meta UMA "
                        "MLIP, default; needs HF_TOKEN) or 'tblite' (GFN-xTB; no "
                        "HF_TOKEN, ~5x faster/step on CPU). A non-default engine "
                        "tags outputs, e.g. <stem>_scan_tblite.csv / "
                        "<stem>_displaced_tblite.xyz / <stem>_stations_tblite/, "
                        "so the engines coexist on disk.")
    p.add_argument("--method", default="GFN2-xTB",
                   help="tblite method (default GFN2-xTB; also GFN1-xTB, GFN0-xTB, "
                        "CEH). Ignored for --engine uma.")
    # scan options
    p.add_argument("--scan", dest="scan", action="store_true", default=True,
                   help="run the relaxed stability-vs-position scan (default on)")
    p.add_argument("--no-scan", dest="scan", action="store_false",
                   help="skip the scan; only write the displaced-isomer XYZ "
                        "(forces --place-rigid)")
    p.add_argument("--dump-stations", dest="dump_stations",
                   action="store_true", default=True,
                   help="write every station's relaxed geometry to "
                        "output_files/<stem>_stations/d{:+.2f}.xyz so "
                        "vib_stations.py can run constrained partial-Hessian "
                        "free energies on the wells/saddles. ON by default; "
                        "chain scans only (rigid --no-scan-chain keeps no "
                        "per-station geometry, so nothing is written).")
    p.add_argument("--no-dump-stations", dest="dump_stations",
                   action="store_false",
                   help="skip writing the per-station geometries "
                        "(vib_stations.py will then have nothing to read).")
    p.add_argument("--scan-grid", type=float, default=SCAN_GRID,
                   help=f"scan spacing in A (default {SCAN_GRID})")
    p.add_argument("--scan-pad", type=float, default=SCAN_PAD,
                   help=f"extend the rigid (--no-scan-chain) scan past each "
                        f"stopper in A (default {SCAN_PAD}); the chain scan "
                        f"ignores this and walks until the tip cutoff")
    p.add_argument("--scan-fmax", type=float, default=SCAN_FMAX,
                   help=f"force tolerance for the per-point UMA relax in eV/A "
                        f"(default {SCAN_FMAX})")
    p.add_argument("--scan-steps", type=int, default=SCAN_STEPS,
                   help=f"max relax steps per grid point (default {SCAN_STEPS})")
    p.add_argument("--production", action="store_true",
                   help="reporting-quality scan: raise --scan-steps to a floor of "
                        "300 so the hard threading-event stations (wheel popping "
                        "through a stopper) actually reach fmax instead of "
                        "hitting the step cap with an unreliable energy. Only "
                        "the 1-3 threading points use the extra steps; the rest "
                        "converge in <100, so the cost is modest.")
    p.add_argument("--scan-emax", type=float, default=SCAN_EMAX,
                   help="clip the plot (not CSV) at min+this kcal/mol; default no clip")
    p.add_argument("--rate-temp", type=float, default=RATE_TEMP,
                   help=f"temperature (K) for the Eyring rate estimates "
                        f"(default {RATE_TEMP:.0f}); estimate only")
    p.add_argument("--no-rates", dest="rates", action="store_false", default=True,
                   help="skip the Eyring rate-estimate block in the scan summary")
    p.add_argument("--scan-walk-emax", type=float, default=WALK_EMAX,
                   help=f"kcal/mol above the scan min at which a chain sweep "
                        f"stops (default {WALK_EMAX}) -- past the last well, "
                        f"into the tip wall")
    p.add_argument("--scan-walk-contact", type=float, default=WALK_CONTACT,
                   help=f"A: stop a chain sweep when min_contact drops below "
                        f"this (default {WALK_CONTACT}) -- a deep tip clash")
    p.add_argument("--scan-chain", dest="scan_chain", action="store_true",
                   default=True,
                   help="relaxed-chain scan: two monotonic outward sweeps from "
                        "the central minimum, each point seeded from the "
                        "previous relaxed geometry so the wheel threads through "
                        "the stopper instead of slamming into it from a fresh "
                        "rigid clash, and each sweep walks until the tip "
                        "cutoff so every well on the side is mapped (default on; "
                        "needed to converge past a stopper on CPU)")
    p.add_argument("--no-scan-chain", dest="scan_chain", action="store_false",
                   help="use the legacy rigid fresh-start scan (one left-to-"
                        "right pass, fresh rigid placement per station); cannot "
                        "map past a stopper, so no side wells are found")
    p.add_argument("--scan-symmetric", dest="scan_symmetric", action="store_true",
                   default=True,
                   help="symmetric (mirror-seeded) chain scan (default on with "
                        "--scan-chain): sweep only +u, then build the -u side by "
                        "mirroring each +d relaxed geometry through the rod-center "
                        "plane and re-relaxing. Both rods are end-to-end symmetric, "
                        "so this removes the two-sweep hysteresis that otherwise "
                        "lopsides the landscape (unequal mirror-side wells/"
                        "barriers). d=0 is the rod center. A rod conformer pre-"
                        "search (--scan-rod-conformers) seeds d=0 from the lowest-"
                        "MMFF RDKit conformer instead of the fixed-seed single embed.")
    p.add_argument("--no-scan-symmetric", dest="scan_symmetric",
                   action="store_false",
                   help="use the legacy two-sweep chain scan (independent +u and "
                        "-u sweeps) -- recovers the asymmetric hysteresis result, "
                        "useful as a comparison/regression check")
    p.add_argument("--scan-sym-tol", type=float, default=SYM_TOL,
                   help=f"kcal/mol: |E(+d) - E(-d)| above this re-relaxes the "
                        f"higher side seeded from the mirror of the lower one "
                        f"(agreement loop; default {SYM_TOL}). Rarely fires -- the "
                        f"mirror seed is already a stationary point.")
    p.add_argument("--scan-sym-iters", type=int, default=SYM_ITERS,
                   help=f"max passes of the symmetry agreement loop (default "
                        f"{SYM_ITERS})")
    p.add_argument("--scan-smooth-pts", type=int, default=SMOOTH_PTS,
                   help=f"moving-average window in POINTS for well detection "
                        f"(default {SMOOTH_PTS}, odd, clamped >= 3). Point-count, "
                        f"not A, so it shrinks with the grid and resolves the "
                        f"~0.4-A stopper wells at fine grids (~1.25 A at 0.25, "
                        f"0.5 A at 0.1) that an A-fixed 1.25 A window smears. "
                        f"Raise to suppress noisier rods, lower to resolve "
                        f"narrower wells. Only affects detection -- well depth "
                        f"comes from the raw snapped floor.")
    p.add_argument("--scan-rod-conformers", type=int, default=ROD_CONFORMERS,
                   help=f"number of RDKit rod conformers to generate + MMFF-rank "
                        f"for the symmetric d=0 seed (default {ROD_CONFORMERS}; "
                        f"avoids the fixed-seed single-embed bias). 0 = use the "
                        f"input rod as-is.")
    p.add_argument("--no-scan-rod-conformers", dest="scan_rod_conformers",
                   action="store_const", const=0,
                   help="use the input rod geometry as the d=0 seed (skip the "
                        "conformer pre-search)")
    args = p.parse_args()

    stem = resolve_stem(args.input)
    out_file = args.out or out_path(stem, "displaced", "xyz", engine=args.engine)
    smiles_path = args.smiles or default_smiles(stem)

    atoms = read(args.input)
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    rod_n, wheel_n = fragment_counts(smiles_path)
    assert len(symbols) == rod_n + wheel_n, \
        f"atom count {len(symbols)} != rod {rod_n} + wheel {wheel_n} " \
        f"(from {smiles_path}); must match the input geometry"
    rod_pos = pos[:rod_n]
    wheel_pos = pos[rod_n:]

    u = rod_axis(rod_pos)
    proj = (rod_pos - rod_pos.mean(axis=0)) @ u
    rod_len = float(proj.max() - proj.min())

    right = max_slide(rod_pos, wheel_pos, u)
    left = max_slide(rod_pos, wheel_pos, -u)
    print(f"rod atoms: {rod_n}  wheel atoms: {wheel_n}  "
          f"rod length along axis: {rod_len:.2f} A")
    print(f"clash-free slide before first rigid stopper:  +right = {right:.2f} A,  "
          f"-left = {left:.2f} A  (advisory; the chain scan threads past these)")

    place_rigid = args.place_rigid or (not args.scan)
    wells = None
    d_min = None
    ds = energies = contacts = None

    # --production raises the per-station relax-step floor so the hard
    # threading-event stations converge (default 200 is for quick exploration;
    # a NOT-converged station reports an unreliable energy).
    scan_steps = max(args.scan_steps, 300) if args.production else args.scan_steps

    # ---- relaxed stability-vs-position scan ----
    if args.scan:
        mode = ("symmetric" if (args.scan_chain and args.scan_symmetric)
                else "chain" if args.scan_chain else "rigid")
        print(f"scan: grid={args.scan_grid} A fmax={args.scan_fmax} eV/A "
              f"steps={scan_steps}  mode={mode}  engine={args.engine}"
              f"{'  [production]' if args.production else ''}", flush=True)
        if args.scan_chain and args.scan_symmetric:
            ds, energies, contacts, pos_by_d, converged = run_scan_chain_sym(
                symbols, pos, rod_n, u, smiles_path,
                grid=args.scan_grid, fmax=args.scan_fmax, steps=scan_steps,
                walk_emax=args.scan_walk_emax,
                walk_contact=args.scan_walk_contact,
                rod_conformers=args.scan_rod_conformers,
                sym_tol=args.scan_sym_tol, sym_iters=args.scan_sym_iters,
                engine=args.engine, method=args.method)
        elif args.scan_chain:
            ds, energies, contacts, pos_by_d, converged = run_scan_chain(
                symbols, pos, rod_n, u, smiles_path,
                grid=args.scan_grid, fmax=args.scan_fmax, steps=scan_steps,
                walk_emax=args.scan_walk_emax,
                walk_contact=args.scan_walk_contact,
                engine=args.engine, method=args.method)
        else:
            ds, energies, contacts, pos_by_d, converged = run_scan(
                symbols, pos, rod_n, u, left, right, smiles_path,
                grid=args.scan_grid, pad=args.scan_pad,
                fmax=args.scan_fmax, steps=scan_steps,
                engine=args.engine, method=args.method)
        wells, d_min, e_min = detect_wells(ds, energies, args.scan_grid,
                                           pos_by_d, converged,
                                           smooth_pts=args.scan_smooth_pts)
        scan_csv = out_path(stem, "scan", "csv", engine=args.engine)
        write_scan_csv(scan_csv, ds, energies, contacts, e_min=e_min)
        print(f"scan: {len(ds)} points  global min at d={d_min:+.2f} A "
              f"(E={e_min:.4f} eV)")
        if wells:
            print(f"detected {len(wells)} well(s) (barrier > {BARRIER_MIN} "
                  f"kcal/mol from the global min):")
            for w in wells:
                print(f"  {w['side']:5s} well  d={w['d']:+.2f} A  "
                      f"E_rel=+{w['rel_kcal']:.2f} kcal/mol  "
                      f"barrier={w['barrier_kcal']:.1f} kcal/mol")
        else:
            print("no wells detected (landscape monotonic past the barrier); "
                  "placement will fall back to the rigid stopper wall")
        if args.rates:
            print_rate_estimates(wells, temperature=args.rate_temp)
        print(f"wrote {scan_csv}", flush=True)

        # Optional: persist every station's relaxed geometry so vib_stations.py
        # can run constrained partial-Hessian free energies on the wells/saddles.
        # pos_by_d is None on the rigid --no-scan-chain path (no per-station
        # geometry), so guard it. Files are keyed by d at 2 decimals to match
        # vib_stations' lookup. The dir is engine-tagged (rot2_stations/ for UMA,
        # rot2_stations_tblite/ for tblite) so the two engines' station geometries
        # don't collide; vib_stations.py is UMA-only today and looks up the
        # untagged dir, so it continues to find UMA stations unchanged.
        if args.dump_stations and pos_by_d is not None:
            import os as _os
            st_dir = _os.path.join(OUTPUT_DIR,
                                   f"{stem}_stations{engine_tag(args.engine)}")
            _os.makedirs(st_dir, exist_ok=True)
            for d, p in pos_by_d.items():
                st_path = _os.path.join(st_dir, f"d{d:+.2f}.xyz")
                write_plain_xyz(st_path, symbols, p,
                                comment=f"station d={d:+.2f} A  E={energies[np.argmin(np.abs(np.asarray(ds) - d))]:.6f} eV")
            print(f"dumped {len(pos_by_d)} station geometries to {st_dir}/",
                  flush=True)

    # ---- placement: at a detected well, or rigid stopper-wall fallback ----
    chosen = None
    if not place_rigid and wells:
        if args.side == "left":
            cands = [w for w in wells if w["side"] == "left"]
            chosen = max(cands, key=lambda w: abs(w["d"])) if cands else None
        elif args.side == "right":
            cands = [w for w in wells if w["side"] == "right"]
            chosen = max(cands, key=lambda w: abs(w["d"])) if cands else None
        elif args.side == "deeper":
            chosen = min(wells, key=lambda w: w["e"])
        else:  # farther (default)
            chosen = max(wells, key=lambda w: abs(w["d"]))
        if chosen is None:
            print(f"--side {args.side}: no well on that side; falling back to "
                  f"rigid placement")

    if chosen is not None:
        # The well geometry is already scan-relaxed under the wheel-rigid /
        # rod-endpoint-anchored constraint set, so it is a valid MD start as-is
        # (run_md.py does not re-relax its input). Center it for a clean XYZ.
        new_pos = chosen["positions"].copy() - chosen["positions"].mean(axis=0)
        write_plain_xyz(
            out_file, symbols, new_pos,
            comment=f"rotaxane at {chosen['side']} well d={chosen['d']:+.2f} A; "
                    f"E_rel=+{chosen['rel_kcal']:.2f} kcal/mol; "
                    f"barrier={chosen['barrier_kcal']:.1f} kcal/mol "
                    f"(scan-relaxed, MD-ready)")
        place_d_signed = chosen["d"]
        place_side = chosen["side"]
        print(f"placed wheel at {chosen['side']} well: d={chosen['d']:+.2f} A "
              f"(E_rel=+{chosen['rel_kcal']:.2f} kcal/mol, "
              f"barrier={chosen['barrier_kcal']:.1f} kcal/mol)")
        print(f"wrote {out_file}  (scan-relaxed; run_md.py can use it directly)")
    else:
        # Rigid stopper-wall placement (legacy / fallback). The resulting geometry
        # is strained, so relax it with optimize_uma.py before run_md.py.
        if args.side in ("left", "right"):
            direction = u if args.side == "right" else -u
            travel = right if args.side == "right" else left
        else:  # farther/deeper with no wells -> farther rigid wall
            if right >= left:
                direction, travel = u, right
                args.side = "right"
            else:
                direction, travel = -u, left
                args.side = "left"
        d_place = max(0.0, travel - args.margin)
        wheel_new = wheel_pos + direction * d_place
        placed_min = min_distance(rod_pos, wheel_new)
        print(f"rigid placement: side={args.side}  wheel at {d_place:.2f} A "
              f"(travel {travel:.2f} minus margin {args.margin}); "
              f"closest rod-wheel contact = {placed_min:.3f} A")
        new_pos = np.vstack([rod_pos, wheel_new])
        new_pos = new_pos - new_pos.mean(axis=0)
        write_plain_xyz(
            out_file, symbols, new_pos,
            comment=f"rotaxane wheel displaced {args.side} {d_place:.2f} A along "
                    f"rod axis; stopper at {travel:.2f} A; closest contact "
                    f"{placed_min:.3f} A (RIGID -- relax with optimize_uma.py "
                    f"before run_md.py)")
        place_d_signed = d_place if args.side == "right" else -d_place
        place_side = args.side
        print(f"wrote {out_file}  (RIGID; relax with optimize_uma.py next)")

    # ---- scan PNG (after placement so the placed marker is on it) ----
    if args.scan:
        scan_png = out_path(stem, "scan", "png", engine=args.engine)
        plot_scan(ds, energies, contacts, left, right, place_d_signed,
                  place_side, scan_png, emax=args.scan_emax,
                  wells=wells, d_min=d_min, e_min=e_min, engine=args.engine)
        print(f"wrote {scan_png}")


if __name__ == "__main__":
    main()