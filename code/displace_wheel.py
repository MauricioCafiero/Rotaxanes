#!/usr/bin/env python
"""Slide the wheel along the rod: scan stability vs position, and place an
extreme-isomer starting geometry for MD.

Reads the UMA-relaxed rotaxane (default <stem>_relaxed.xyz), finds the rod's
long axis by PCA, and translates the wheel along +/- that axis. Two things
happen:

1. **Stability scan** (the main output): the wheel is slid rigidly across the
   clash-free travel window on a grid and a UMA single-point energy is
   evaluated at each position, producing `<stem>_scan.png` (energy vs
   displacement) and `<stem>_scan.csv`. UMA is a smooth ML potential, so on the
   relaxed input this gives a finite shuttle landscape with real minima and
   barriers as the wheel passes over the rod's phenyl/CF3 features -- unlike a
   classical force field, whose 1/r^12 vdW term diverges at the transient close
   contacts and swamps the subtle wells. The UMA model is loaded once and
   reused per point (a single-point energy is quick even on CPU). Needs
   HF_TOKEN. Use `--no-scan` to skip.

2. **Extreme placement** (the MD starting geometry): the wheel is placed as far
   as possible along the chosen side before running into a rod stopper, and
   written to `<stem>_displaced.xyz` for `run_md.py`. Relax that next with
   `optimize_uma.py --input <stem>_displaced.xyz`.

The rod is bumpy (phenyl/CF3 groups), so as the wheel slides its closest
contact dips and recovers as features pass through the ring. A real stopper is
therefore a *sustained* overlap: the minimum rod-wheel distance stays below a
floor over a width of several angstroms. We scan outward and stop at the first
such sustained wall; the wheel is placed just before it (minus a safety margin).
"""

import argparse
import csv

import numpy as np
from ase.io import read
from rdkit import Chem

from build_rotaxane import read_smiles
from rotaxane_paths import resolve_stem, out_path, default_smiles

DEFAULT_IN = out_path("rot_smiles", "relaxed", "xyz")

FLOOR = 1.0        # A; closer than this between a rod-wheel pair = overlap
WALL_WIDTH = 3.0   # A; a stopper is an overlap sustained over this width
SCAN_STEP = 0.05   # A scan increment for stopper detection
MARGIN = 0.3       # A safety back-off from the wall
MAX_SLIDE = 25.0   # A search limit (rod is ~28 A long)

SCAN_GRID = 0.5    # A spacing of the stability-vs-position energy scan (a
                   # relaxed UMA minimisation per grid point; 0.5 A keeps a CPU
                   # scan quick, pass --scan-grid 0.25/0.1 for a finer landscape)
SCAN_PAD = 0.0     # A: extend the scan past each stopper (0 = scan only the
                   # clash-free accessible window, where the landscape lives)
SCAN_FMAX = 0.5    # eV/A: loose force tolerance for the per-point relaxation --
                   # just enough to relieve bad sterics, not a full minimisation
SCAN_STEPS = 20    # max relax steps per grid point (the first few do most of the
                   # clash-relief; cap keeps a CPU scan quick)
SCAN_EMAX = None   # eV: clip the plot (not the CSV) at min + this; None = no
                   # clip. Set (e.g. --scan-emax 5) only if a stray point blows
                   # up the y-axis.


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


def sustained_wall(rod_pos, wheel_pos, direction, d0):
    """True if the overlap below FLOOR is sustained over WALL_WIDTH starting d0."""
    for dd in np.arange(d0, d0 + WALL_WIDTH + SCAN_STEP, SCAN_STEP):
        w = wheel_pos + direction * dd
        if min_distance(rod_pos, w) >= FLOOR:
            return False
    return True


def max_slide(rod_pos, wheel_pos, direction):
    """Largest shift along `direction` before the first sustained stopper wall."""
    d = 0.0
    while d <= MAX_SLIDE:
        if sustained_wall(rod_pos, wheel_pos, direction, d):
            return max(0.0, d - SCAN_STEP)
        d += SCAN_STEP
    return MAX_SLIDE


def write_plain_xyz(path, symbols, pos, comment=""):
    with open(path, "w") as fh:
        fh.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, pos):
            fh.write(f"{s:<2} {x: .6f} {y: .6f} {z: .6f}\n")


# --------------------------------------------------------------------------- #
# Stability-vs-position scan (relaxed UMA per wheel position)
# --------------------------------------------------------------------------- #
def run_scan(symbols, pos0, rod_n, u, left, right, smiles_path,
             grid=SCAN_GRID, pad=SCAN_PAD, fmax=SCAN_FMAX, steps=SCAN_STEPS):
    """Relaxed UMA scan of the wheel along the rod.

    At each grid point the wheel is placed rigidly at `wheel0 + u*d` (station d
    along the rod), then UMA relaxes the structure with:
      - the wheel held RIGID and fixed at station d (FixAtoms), so it stays
        mechanically threaded on the rod and cannot pop off sideways, and
      - the rod's two endpoint atoms anchored (FixAtoms), so the rod can flex
        internally to relieve close contacts but cannot translate/escape.
    So the only degrees of freedom are the rod's internal flex (phenyl/CF3
    groups rotating away from the wheel) -- exactly the clash-relief we want,
    while the scan coordinate (wheel station along the rod) stays fixed. A loose
    convergence (fmax, steps) is used -- just enough to push apart bad sterics,
    not a full minimisation -- so each point is quick even on CPU.

    Returns (ds, energies, min_contacts). `ds` are signed displacements along
    `u` (positive = +u direction); `energies` are the relaxed UMA potential
    energies (eV); `min_contacts` are the closest rod-wheel distances at the
    *relaxed* geometry.
    """
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS
    from optimize_uma import (
        MODEL, TASK, VACUUM, get_hf_token, pick_device, read_charge_spin,
    )
    from fairchem.core import pretrained_mlip, FAIRChemCalculator

    get_hf_token()
    charge, spin = read_charge_spin(smiles_path)
    device = pick_device()
    print(f"scan: UMA relaxed  model={MODEL} task={TASK} device={device} "
          f"charge={charge} spin={spin}  fmax={fmax} eV/A  steps={steps}")

    atoms = Atoms(symbols=symbols, positions=np.asarray(pos0, dtype=float))
    atoms.set_pbc(False)
    atoms.center(vacuum=VACUUM)  # large non-periodic box; translation-invariant
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    predictor = pretrained_mlip.get_predict_unit(MODEL, device=device)
    atoms.calc = FAIRChemCalculator(predictor, task_name=TASK)

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
        # wheel centroid along u relative to rod centroid (should track d;
        # small offset = the rod centroid shifting as the rod flexes)
        wc_u = float((relaxed[rod_n:].mean(axis=0) - relaxed[:rod_n].mean(axis=0)) @ u)
        print(f"  scan {k + 1}/{len(ds)}: d={d:+.2f} A  "
              f"E={energies[k]:.4f} eV  min_contact={contacts[k]:.2f} A  "
              f"wheel_u={wc_u:+.2f} A  (relax steps={opt.get_number_of_steps()})",
              flush=True)
    return ds, energies, contacts


def plot_scan(ds, energies, contacts, left, right, place_d, place_side,
              out_png, emax=SCAN_EMAX):
    """Write the energy-vs-position PNG. Energy is plotted relative to its min;
    if `emax` is set, the plot (not the CSV) is clipped at min + emax so a
    stray point doesn't flatten the landscape. Stopper walls and the placed
    extreme are marked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel = energies - energies.min()
    disp = np.clip(rel, None, emax) if emax is not None else rel

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ds, disp, "o-", ms=3, lw=1.0, color="tab:blue",
            label="UMA relaxed energy, rel. to min")
    ax.axvline(right, color="tab:red", ls="--", lw=1.0, label=f"+stopper ({right:.2f} A)")
    ax.axvline(-left, color="tab:red", ls=":", lw=1.0, label=f"-stopper ({-left:.2f} A)")
    if place_d is not None:
        ax.axvline(place_d, color="tab:green", lw=1.2,
                   label=f"placed {place_side} ({place_d:.2f} A)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("wheel displacement along rod axis (A)")
    ax.set_ylabel("energy - E_min  (UMA, eV)")
    ax.set_title("Rotaxane shuttle stability vs wheel position (relaxed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def write_scan_csv(path, ds, energies, contacts):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["displacement_A", "energy_UMA_eV", "min_contact_A"])
        for d, e, c in zip(ds, energies, contacts):
            w.writerow([f"{d:.4f}", f"{e:.6f}", f"{c:.4f}"])


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_IN,
                   help="relaxed rotaxane geometry (default: <stem>_relaxed.xyz). "
                        "Outputs are named from this file's stem.")
    p.add_argument("--side", choices=["left", "right", "farther"],
                   default="farther",
                   help="left/right = -/+ rod axis; farther (default) = the "
                        "side with the larger clash-free travel")
    p.add_argument("--margin", type=float, default=MARGIN,
                   help="A back-off from the stopper wall (default 0.3)")
    p.add_argument("--out", default=None,
                   help="displaced-isomer XYZ (default: <stem>_displaced.xyz)")
    p.add_argument("--smiles", default=None,
                   help="rod:/wheel: file for atom counts + charge/spin "
                        "(default: <stem>.txt matching the input)")
    # scan options
    p.add_argument("--scan", dest="scan", action="store_true", default=True,
                   help="run the relaxed stability-vs-position scan (default on)")
    p.add_argument("--no-scan", dest="scan", action="store_false",
                   help="skip the scan; only write the displaced-isomer XYZ")
    p.add_argument("--scan-grid", type=float, default=SCAN_GRID,
                   help=f"scan spacing in A (default {SCAN_GRID})")
    p.add_argument("--scan-pad", type=float, default=SCAN_PAD,
                   help=f"extend scan past each stopper in A (default {SCAN_PAD})")
    p.add_argument("--scan-fmax", type=float, default=SCAN_FMAX,
                   help=f"loose force tolerance for the per-point UMA relax in "
                        f"eV/A (default {SCAN_FMAX} -- just enough to relieve "
                        f"bad sterics, not a full minimisation)")
    p.add_argument("--scan-steps", type=int, default=SCAN_STEPS,
                   help=f"max relax steps per grid point (default {SCAN_STEPS})")
    p.add_argument("--scan-emax", type=float, default=SCAN_EMAX,
                   help="clip the plot (not CSV) at min+this eV; default no clip")
    args = p.parse_args()

    stem = resolve_stem(args.input)
    out_file = args.out or out_path(stem, "displaced", "xyz")
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
    print(f"clash-free slide before stopper:  +right = {right:.2f} A,  "
          f"-left = {left:.2f} A")

    if args.side == "right":
        direction, travel, side = u, right, "right"
    elif args.side == "left":
        direction, travel, side = -u, left, "left"
    else:  # farther
        if right >= left:
            direction, travel, side = u, right, "right"
        else:
            direction, travel, side = -u, left, "left"

    d_place = max(0.0, travel - args.margin)
    wheel_new = wheel_pos + direction * d_place
    placed_min = min_distance(rod_pos, wheel_new)
    print(f"selected side: {side}  ->  wheel placed at {d_place:.2f} A "
          f"(travel {travel:.2f} minus margin {args.margin}); "
          f"closest rod-wheel contact there = {placed_min:.3f} A")

    new_pos = np.vstack([rod_pos, wheel_new])
    new_pos = new_pos - new_pos.mean(axis=0)
    write_plain_xyz(
        out_file, symbols, new_pos,
        comment=f"rotaxane wheel displaced {side} {d_place:.2f} A along rod "
                f"axis; stopper at {travel:.2f} A; closest contact "
                f"{placed_min:.3f} A",
    )
    print(f"wrote {out_file}")

    # ---- relaxed stability-vs-position scan (UMA) ----
    if args.scan:
        print(f"scan: grid={args.scan_grid} A pad={args.scan_pad} A "
              f"fmax={args.scan_fmax} eV/A steps={args.scan_steps}  "
              f"range=[{-left - args.scan_pad:.2f}, {right + args.scan_pad:.2f}] A")
        ds, energies, contacts = run_scan(
            symbols, pos, rod_n, u, left, right, smiles_path,
            grid=args.scan_grid, pad=args.scan_pad,
            fmax=args.scan_fmax, steps=args.scan_steps)
        scan_png = out_path(stem, "scan", "png")
        scan_csv = out_path(stem, "scan", "csv")
        write_scan_csv(scan_csv, ds, energies, contacts)
        plot_scan(ds, energies, contacts, left, right,
                  d_place if side == "right" else -d_place, side,
                  scan_png, emax=args.scan_emax)
        i_min = int(np.argmin(energies))
        acc = energies[(ds >= -left) & (ds <= right)]
        barrier = float(acc.max() - acc.min()) if acc.size else float("nan")
        print(f"scan: {len(ds)} points  min energy at d={ds[i_min]:+.2f} A "
              f"(E_rel=0); accessible-window barrier={barrier:.3f} eV")
        print(f"wrote {scan_png}  and  {scan_csv}")


if __name__ == "__main__":
    main()